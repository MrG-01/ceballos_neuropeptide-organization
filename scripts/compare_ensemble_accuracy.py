import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import zscore, spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet
from pyls import behavioral_pls
from utils import get_centroids

# Ensure figs and results directories exist
os.makedirs('figs', exist_ok=True)
os.makedirs('results', exist_ok=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         LOAD DATA
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Loading data...")
ns = pd.read_csv('./data/neurosynth_Schaefer400_TianS4.csv', index_col=0)
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
lut = pd.read_csv('data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv', index_col=0)
overview = pd.read_csv('data/receptor_overview.csv')

# Build mapping of gene to family
modeled_genes = receptor_genes.columns.tolist()
overview_filtered = overview[overview['gene'].isin(modeled_genes)].copy()
overview_filtered['family'] = overview_filtered['family'].fillna(overview_filtered['gene'])
overview_filtered['family'] = overview_filtered['family'].replace('None', np.nan)
overview_filtered['family'] = overview_filtered['family'].fillna(overview_filtered['gene'])
gene_to_family = dict(zip(overview_filtered['gene'], overview_filtered['family']))

# Group and average individual receptor genes by neuropeptide family (warning-free style)
family_genes = receptor_genes.copy()
family_genes.columns = [gene_to_family.get(c, c) for c in family_genes.columns]
family_genes = family_genes.T.groupby(level=0).mean().T

X = zscore(family_genes.values, ddof=1)
Y = zscore(ns.values, ddof=1)

# Centroids and distance matrix for G-PLS
img = nib.load('data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz')
labels = np.unique(img.get_fdata())[1:]
centroids = get_centroids(img, labels=labels)
distance = squareform(pdist(centroids))

# Load LUT for structural labels
lut_matched = lut.iloc[:len(centroids)]
is_cortex = (lut_matched['structure'] == 'cortex').astype(float).values
radial_dist = np.sqrt((centroids ** 2).sum(axis=1))

# Load Structural Connectivity (SC) and pad for Hypothalamus
sc_raw = np.load('data/template_parc-Schaefer400_TianS4_desc-SC.npy')
N = len(centroids)  # 455
sc_padded = np.zeros((N, N))
sc_padded[:454, :454] = sc_raw

# Estimate connection weights for Hypothalamus using a spatial Gaussian distance kernel
D_hth = distance[454, :]
sc_hth = np.exp(- (D_hth ** 2) / (2 * 15.0 ** 2))
sc_hth[454] = 0.0
sc_padded[454, :] = sc_hth
sc_padded[:, 454] = sc_hth

# Load Functional Connectivity (FC) and pad for Hypothalamus
fc_raw = np.load('data/template_parc-Schaefer400_TianS4_desc-FC.npy')
fc_padded = np.zeros((N, N))
fc_padded[:454, :454] = np.abs(fc_raw)
closest_subcortex_idx = np.argsort(D_hth)[:4]
closest_subcortex_idx = closest_subcortex_idx[closest_subcortex_idx < 454][:3]
fc_padded[454, :454] = np.abs(fc_raw[closest_subcortex_idx, :]).mean(axis=0)
fc_padded[:454, 454] = fc_padded[454, :454]
fc_padded[454, 454] = 0.0

# Calculate node strengths for gating features
sc_strength = sc_padded.sum(axis=1)
fc_strength = fc_padded.sum(axis=1)

# Construct enriched spatial + connectomic features
spatial_features = np.column_stack([centroids, radial_dist, is_cortex, sc_strength, fc_strength])

# Normalize FC adjacency matrix: D^{-1/2} * (A + I) * D^{-1/2}
A_tilde = fc_padded + np.eye(N)
deg = A_tilde.sum(axis=1)
deg_inv_sqrt = 1.0 / np.sqrt(deg)
deg_inv_sqrt[np.isinf(deg_inv_sqrt) | np.isnan(deg_inv_sqrt)] = 0.0
D_inv_sqrt = np.diag(deg_inv_sqrt)
A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt

# Convert to tensors
adj_tensor = torch.tensor(A_hat, dtype=torch.float32)
X_tensor = torch.tensor(X, dtype=torch.float32)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         MODEL CLASS DEFINITIONS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class GraphRegularizedPLS:
    def __init__(self, lam=1.0):
        self.lam = lam

    def _compute_laplacian(self, dist_matrix, sigma=15.0):
        W = np.exp(- (dist_matrix ** 2) / (2 * sigma ** 2))
        np.fill_diagonal(W, 0)
        deg = W.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            deg_inv_sqrt = 1.0 / np.sqrt(deg)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt) | np.isnan(deg_inv_sqrt)] = 0
        D_inv_sqrt = np.diag(deg_inv_sqrt)
        L_norm = np.eye(dist_matrix.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
        return L_norm

    def fit(self, X, Y, dist_matrix):
        L = self._compute_laplacian(dist_matrix)
        Px = np.eye(X.shape[1]) + self.lam * (X.T @ L @ X)
        Py = np.eye(Y.shape[1]) + self.lam * (Y.T @ L @ Y)
        
        from scipy.linalg import pinv
        Px_inv = pinv(Px)
        Py_inv = pinv(Py)

        u = pinv(Py) @ Y.T @ X[:, [0]]
        u /= np.linalg.norm(u)
        
        for _ in range(100):
            v = Px_inv @ (X.T @ Y) @ u
            v_norm = np.sqrt(v.T @ Px @ v)
            if v_norm > 0:
                v /= v_norm
            
            u_new = Py_inv @ (Y.T @ X) @ v
            u_norm = np.sqrt(u_new.T @ Py @ u_new)
            if u_norm > 0:
                u_new /= u_norm
            
            if np.linalg.norm(u - u_new) < 1e-6:
                u = u_new
                break
            u = u_new

        v = Px_inv @ (X.T @ Y) @ u
        v_norm = np.sqrt(v.T @ Px @ v)
        if v_norm > 0:
            v /= v_norm

        self.x_weights_ = v
        self.y_weights_ = u
        return self

    def transform(self, X_test, Y_test=None):
        t_scores = X_test @ self.x_weights_
        if Y_test is not None:
            u_scores = Y_test @ self.y_weights_
            return t_scores, u_scores
        return t_scores

class BrainGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1, dropout=0.5):
        super(BrainGCN, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj):
        h = adj @ self.linear1(x)
        h = F.relu(h)
        h = self.dropout(h)
        out = adj @ self.linear2(h)
        return out

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         EVALUATION LOOP
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_splits = 100
test_size = 0.7

metrics = {
    "Split": [],
    "PLSC_Corr": [],
    "PLSC_MSE": [],
    "PLSC_RMSE": [],
    "GPLS_Corr": [],
    "GPLS_MSE": [],
    "GPLS_RMSE": [],
    "Ensemble_Corr": [],
    "Ensemble_MSE": [],
    "Ensemble_RMSE": []
}

print(f"Running {n_splits}-fold cross-validation to compare accuracy metrics...")

for i in range(n_splits):
    if (i+1) % 10 == 0:
        print(f"Completed {i+1}/{n_splits} splits...")

    train_idx, test_idx = train_test_split(np.arange(X.shape[0]), test_size=test_size, random_state=i)
    Xtrain, Ytrain = X[train_idx], Y[train_idx]
    Xtest, Ytest = X[test_idx], Y[test_idx]
    dist_train = distance[train_idx][:, train_idx]

    feats_train = spatial_features[train_idx]
    feats_test = spatial_features[test_idx]

    # 1. Fit PLSC
    try:
        plsc = behavioral_pls(Xtrain, Ytrain, n_boot=0, n_perm=0, test_split=0)
        t_plsc_test = Xtest @ plsc["x_weights"][:, 0]
        u_plsc_test = Ytest @ plsc["y_weights"][:, 0]
        
        # Align sign
        if pearsonr(t_plsc_test, u_plsc_test)[0] < 0:
            t_plsc_test = -t_plsc_test
            
        r_plsc = spearmanr(t_plsc_test, u_plsc_test)[0]
        mse_plsc = np.mean((t_plsc_test - u_plsc_test) ** 2)
        rmse_plsc = np.sqrt(mse_plsc)
    except Exception as e:
        # If PLSC fails on a rare split, skip this split to maintain matched cases
        continue

    # 2. Fit G-PLS
    gpls = GraphRegularizedPLS(lam=1.0).fit(Xtrain, Ytrain, dist_train)
    t_gpls_train, u_gpls_train = gpls.transform(Xtrain, Ytrain)
    t_gpls_test, u_gpls_test = gpls.transform(Xtest, Ytest)
    t_gpls_train, u_gpls_train = t_gpls_train[:, 0], u_gpls_train[:, 0]
    t_gpls_test, u_gpls_test = t_gpls_test[:, 0], u_gpls_test[:, 0]
    
    # Align sign
    if pearsonr(t_gpls_test, u_gpls_test)[0] < 0:
        t_gpls_test = -t_gpls_test
        t_gpls_train = -t_gpls_train

    r_gpls = spearmanr(t_gpls_test, u_gpls_test)[0]
    mse_gpls = np.mean((t_gpls_test - u_gpls_test) ** 2)
    rmse_gpls = np.sqrt(mse_gpls)

    # 3. Fit RF Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=i, n_jobs=-1)
    rf.fit(Xtrain, u_gpls_train)
    u_rf_train = rf.predict(Xtrain)
    u_rf_test = rf.predict(Xtest)
    if pearsonr(u_rf_train, u_gpls_train)[0] < 0:
        u_rf_train, u_rf_test = -u_rf_train, -u_rf_test

    # 4. Fit ElasticNet
    en = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=i, max_iter=2000)
    en.fit(Xtrain, u_gpls_train)
    u_en_train = en.predict(Xtrain)
    u_en_test = en.predict(Xtest)
    if pearsonr(u_en_train, u_gpls_train)[0] < 0:
        u_en_train, u_en_test = -u_en_train, -u_en_test

    # 5. Fit BrainGCN
    y_target = np.zeros((N, 1))
    y_target[train_idx, 0] = u_gpls_train
    y_target_tensor = torch.tensor(y_target, dtype=torch.float32)

    model_gcn = BrainGCN(in_dim=family_genes.shape[1], hidden_dim=64, out_dim=1, dropout=0.5)
    optimizer = torch.optim.Adam(model_gcn.parameters(), lr=0.01, weight_decay=1e-4)

    model_gcn.train()
    for epoch in range(200):
        optimizer.zero_grad()
        pred = model_gcn(X_tensor, adj_tensor)
        loss = F.mse_loss(pred[train_idx], y_target_tensor[train_idx])
        loss.backward()
        optimizer.step()

    model_gcn.eval()
    with torch.no_grad():
        gcn_pred = model_gcn(X_tensor, adj_tensor).numpy().flatten()
    u_gcn_train = gcn_pred[train_idx]
    u_gcn_test = gcn_pred[test_idx]
    if pearsonr(u_gcn_train, u_gpls_train)[0] < 0:
        u_gcn_train, u_gcn_test = -u_gcn_train, -u_gcn_test

    # Prediction errors for training regions
    err_gpls = np.abs(u_gpls_train - t_gpls_train)
    err_rf = np.abs(u_gpls_train - u_rf_train)
    err_en = np.abs(u_gpls_train - u_en_train)
    err_gcn = np.abs(u_gpls_train - u_gcn_train)

    # Best training model index per region
    best_model_idx = np.argmin(np.column_stack([err_gpls, err_rf, err_en, err_gcn]), axis=1)

    # Train gating classifier
    unique_classes = np.unique(best_model_idx)
    if len(unique_classes) > 1:
        gating_net = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=35, random_state=i, n_jobs=-1)
        gating_net.fit(feats_train, best_model_idx)
        pred_weights_raw = gating_net.predict_proba(feats_test)
        pred_weights = np.zeros((len(test_idx), 4))
        for idx, cls in enumerate(gating_net.classes_):
            pred_weights[:, cls] = pred_weights_raw[:, idx]
    else:
        pred_weights = np.zeros((len(test_idx), 4))
        pred_weights[:, unique_classes[0]] = 1.0

    # Dynamic gating combination
    u_dynamic = (pred_weights[:, 0] * u_gpls_test + 
                 pred_weights[:, 1] * u_rf_test + 
                 pred_weights[:, 2] * u_en_test +
                 pred_weights[:, 3] * u_gcn_test)

    # Align ensemble sign to target if necessary (though already aligned as components are aligned)
    if pearsonr(u_dynamic, u_gpls_test)[0] < 0:
        u_dynamic = -u_dynamic

    r_ens = spearmanr(u_dynamic, u_gpls_test)[0]
    mse_ens = np.mean((u_dynamic - u_gpls_test) ** 2)
    rmse_ens = np.sqrt(mse_ens)

    # Append results
    metrics["Split"].append(i)
    metrics["PLSC_Corr"].append(r_plsc)
    metrics["PLSC_MSE"].append(mse_plsc)
    metrics["PLSC_RMSE"].append(rmse_plsc)
    metrics["GPLS_Corr"].append(r_gpls)
    metrics["GPLS_MSE"].append(mse_gpls)
    metrics["GPLS_RMSE"].append(rmse_gpls)
    metrics["Ensemble_Corr"].append(r_ens)
    metrics["Ensemble_MSE"].append(mse_ens)
    metrics["Ensemble_RMSE"].append(rmse_ens)

df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv('results/ensemble_accuracy_comparison.csv', index=False)
print("Accuracy metrics saved to results/ensemble_accuracy_comparison.csv")

# Print Summary Statistics
print("\n" + "="*50)
print("             ACCURACY PERFORMANCE SUMMARY")
print("="*50)
print(f"Model      | Out-of-Sample Corr (Mean ± SD) | RMSE (Mean ± SD)  | MSE (Mean ± SD)")
print("-"*50)
print(f"PLSC       | {df_metrics['PLSC_Corr'].mean():.4f} ± {df_metrics['PLSC_Corr'].std():.4f}      | {df_metrics['PLSC_RMSE'].mean():.4f} ± {df_metrics['PLSC_RMSE'].std():.4f}   | {df_metrics['PLSC_MSE'].mean():.4f} ± {df_metrics['PLSC_MSE'].std():.4f}")
print(f"G-PLS      | {df_metrics['GPLS_Corr'].mean():.4f} ± {df_metrics['GPLS_Corr'].std():.4f}      | {df_metrics['GPLS_RMSE'].mean():.4f} ± {df_metrics['GPLS_RMSE'].std():.4f}   | {df_metrics['GPLS_MSE'].mean():.4f} ± {df_metrics['GPLS_MSE'].std():.4f}")
print(f"Ensemble   | {df_metrics['Ensemble_Corr'].mean():.4f} ± {df_metrics['Ensemble_Corr'].std():.4f}      | {df_metrics['Ensemble_RMSE'].mean():.4f} ± {df_metrics['Ensemble_RMSE'].std():.4f}   | {df_metrics['Ensemble_MSE'].mean():.4f} ± {df_metrics['Ensemble_MSE'].std():.4f}")
print("="*50)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOTTING FIGURES
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Generating plots...")

# Design setup
sns.set_theme(style="whitegrid")
palette_colors = ["#4C72B0", "#55A868", "#C44E52"] # Curated premium palette for PLSC, G-PLS, Ensemble
model_labels = ["PLSC", "G-PLS", "Ensemble"]

# Plot 1: Out-of-Sample Latent Score Correlation
fig1, ax1 = plt.subplots(figsize=(6.5, 5), dpi=300)
df_corr = df_metrics[["PLSC_Corr", "GPLS_Corr", "Ensemble_Corr"]].copy()
df_corr.columns = model_labels

sns.boxplot(data=df_corr, palette=palette_colors, width=0.4, linewidth=1.2, showfliers=False, ax=ax1)
sns.stripplot(data=df_corr, palette=palette_colors, size=3.0, jitter=0.2, alpha=0.4, edgecolor="gray", linewidth=0.5, ax=ax1)
ax1.set_ylabel("Out-of-Sample Latent Score Correlation (Spearman)", fontsize=11, fontweight='bold', labelpad=10)
ax1.set_title("Out-of-Sample Latent Score Correlation", fontsize=13, fontweight='bold', pad=15)
ax1.set_ylim(-0.2, 1.0)
sns.despine(trim=True)
fig1.savefig('figs/ensemble_accuracy_correlation.pdf', bbox_inches='tight')
fig1.savefig('figs/ensemble_accuracy_correlation.png', bbox_inches='tight')
plt.close(fig1)

# Plot 2: Prediction Errors (MSE / RMSE)
fig2, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# Left: MSE
ax_mse = axes[0]
df_mse = df_metrics[["PLSC_MSE", "GPLS_MSE", "Ensemble_MSE"]].copy()
df_mse.columns = model_labels
sns.boxplot(data=df_mse, palette=palette_colors, width=0.4, linewidth=1.2, showfliers=False, ax=ax_mse)
sns.stripplot(data=df_mse, palette=palette_colors, size=3.0, jitter=0.2, alpha=0.4, edgecolor="gray", linewidth=0.5, ax=ax_mse)
ax_mse.set_ylabel("Mean Squared Error (MSE)", fontsize=11, fontweight='bold', labelpad=10)
ax_mse.set_title("Out-of-Sample Mean Squared Error (MSE)", fontsize=12, fontweight='bold', pad=15)
ax_mse.set_ylim(-0.1, max(df_metrics["PLSC_MSE"].max(), df_metrics["GPLS_MSE"].max()) * 1.1)

# Right: RMSE
ax_rmse = axes[1]
df_rmse = df_metrics[["PLSC_RMSE", "GPLS_RMSE", "Ensemble_RMSE"]].copy()
df_rmse.columns = model_labels
sns.boxplot(data=df_rmse, palette=palette_colors, width=0.4, linewidth=1.2, showfliers=False, ax=ax_rmse)
sns.stripplot(data=df_rmse, palette=palette_colors, size=3.0, jitter=0.2, alpha=0.4, edgecolor="gray", linewidth=0.5, ax=ax_rmse)
ax_rmse.set_ylabel("Root Mean Squared Error (RMSE)", fontsize=11, fontweight='bold', labelpad=10)
ax_rmse.set_title("Out-of-Sample Root Mean Squared Error (RMSE)", fontsize=12, fontweight='bold', pad=15)
ax_rmse.set_ylim(-0.1, max(df_metrics["PLSC_RMSE"].max(), df_metrics["GPLS_RMSE"].max()) * 1.1)

for ax in axes:
    sns.despine(ax=ax, trim=True)

plt.tight_layout()
fig2.savefig('figs/ensemble_accuracy_error.pdf', bbox_inches='tight')
fig2.savefig('figs/ensemble_accuracy_error.png', bbox_inches='tight')
plt.close(fig2)

print("Accuracy plots successfully saved to figs/ensemble_accuracy_correlation.* and figs/ensemble_accuracy_error.*")
