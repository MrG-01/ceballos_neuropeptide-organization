import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import zscore, spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from pyls import behavioral_pls
from utils import get_centroids

# Ensure figs directory exists
os.makedirs('figs', exist_ok=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         LOAD DATA
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ns = pd.read_csv('./data/neurosynth_Schaefer400_TianS4.csv', index_col=0)
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
lut = pd.read_csv('data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv', index_col=0)
overview = pd.read_csv('data/receptor_overview.csv')

# Build mapping of gene to family
# Map receptors to their neuropeptide family; treat 'None' or missing as single-gene families
modeled_genes = receptor_genes.columns.tolist()
overview_filtered = overview[overview['gene'].isin(modeled_genes)].copy()
overview_filtered['family'] = overview_filtered['family'].fillna(overview_filtered['gene'])
overview_filtered['family'] = overview_filtered['family'].replace('None', np.nan)
overview_filtered['family'] = overview_filtered['family'].fillna(overview_filtered['gene'])
gene_to_family = dict(zip(overview_filtered['gene'], overview_filtered['family']))

# Group and average individual receptor genes by neuropeptide family
family_genes = receptor_genes.copy()
family_genes.columns = [gene_to_family.get(c, c) for c in family_genes.columns]
family_genes = family_genes.groupby(by=family_genes.columns, axis=1).mean()

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
#                         DYNAMIC STACKING LOOP
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_splits = 100
test_size = 0.7
static_alpha = 0.7
tau = 0.5  # Softmin temperature for gating weights

results = {
    "PLSC (Baseline)": [],
    "G-PLS (Baseline)": [],
    "G-PLS + RF (Static 70/30)": [],
    "G-PLS + RF + EN + GCN (Dynamic Gating)": []
}

# Track gating weights for cortex vs subcortex regions
cortex_weights_accum = []
subcortex_weights_accum = []

print(f"Running {n_splits}-fold cross-validation for Dynamic Gating Stack...")

for i in range(n_splits):
    train_idx, test_idx = train_test_split(np.arange(X.shape[0]), test_size=test_size, random_state=i)
    Xtrain, Ytrain = X[train_idx], Y[train_idx]
    Xtest, Ytest = X[test_idx], Y[test_idx]
    dist_train = distance[train_idx][:, train_idx]

    # MNI Centroids and enriched spatial features for Gating
    coords_train = centroids[train_idx]
    coords_test = centroids[test_idx]
    feats_train = spatial_features[train_idx]
    feats_test = spatial_features[test_idx]

    # 1. Fit PLSC Baseline
    try:
        plsc = behavioral_pls(Xtrain, Ytrain, n_boot=0, n_perm=0, test_split=0)
        t_plsc_test = Xtest @ plsc["x_weights"][:, 0]
        u_plsc_test = Ytest @ plsc["y_weights"][:, 0]
        results["PLSC (Baseline)"].append(spearmanr(t_plsc_test, u_plsc_test)[0])
    except:
        continue

    # 2. Fit G-PLS Baseline
    gpls = GraphRegularizedPLS(lam=1.0).fit(Xtrain, Ytrain, dist_train)
    t_gpls_train, u_gpls_train = gpls.transform(Xtrain, Ytrain)
    t_gpls_test, u_gpls_test = gpls.transform(Xtest, Ytest)
    t_gpls_train, u_gpls_train = t_gpls_train[:, 0], u_gpls_train[:, 0]
    t_gpls_test, u_gpls_test = t_gpls_test[:, 0], u_gpls_test[:, 0]
    results["G-PLS (Baseline)"].append(spearmanr(t_gpls_test, u_gpls_test)[0])

    # 3. Fit Random Forest
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

    # 5. Evaluate Static Blend (G-PLS + RF)
    u_static = static_alpha * t_gpls_test + (1 - static_alpha) * u_rf_test
    results["G-PLS + RF (Static 70/30)"].append(spearmanr(t_gpls_test, u_static)[0])

    # 6. Fit GCN
    u_true_train = Ytrain @ gpls.y_weights_[:, 0]
    y_target = np.zeros((N, 1))
    y_target[train_idx, 0] = u_true_train
    y_target_tensor = torch.tensor(y_target, dtype=torch.float32)

    model = BrainGCN(in_dim=family_genes.shape[1], hidden_dim=64, out_dim=1, dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        pred = model(X_tensor, adj_tensor)
        loss = F.mse_loss(pred[train_idx], y_target_tensor[train_idx])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        gcn_pred = model(X_tensor, adj_tensor).numpy().flatten()
    u_gcn_train = gcn_pred[train_idx]
    u_gcn_test = gcn_pred[test_idx]
    if pearsonr(u_gcn_train, u_true_train)[0] < 0:
        u_gcn_train, u_gcn_test = -u_gcn_train, -u_gcn_test

    # Calculate prediction errors for each base learner
    err_gpls = np.abs(u_true_train - t_gpls_train)
    err_rf = np.abs(u_true_train - u_rf_train)
    err_en = np.abs(u_true_train - u_en_train)
    err_gcn = np.abs(u_true_train - u_gcn_train)

    # Determine the best model for each training region
    best_model_idx = np.argmin(np.column_stack([err_gpls, err_rf, err_en, err_gcn]), axis=1)

    # Train RandomForest gating meta-classifier to predict best model
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

    # Track weights for interpretation (cortex vs subcortex)
    cortex_test_mask = is_cortex[test_idx] == 1.0
    subcortex_test_mask = is_cortex[test_idx] == 0.0
    if cortex_test_mask.any():
        cortex_weights_accum.append(pred_weights[cortex_test_mask].mean(axis=0))
    if subcortex_test_mask.any():
        subcortex_weights_accum.append(pred_weights[subcortex_test_mask].mean(axis=0))

    # Compute dynamically stacked scores for each test region
    u_dynamic = (pred_weights[:, 0] * t_gpls_test + 
                 pred_weights[:, 1] * u_rf_test + 
                 pred_weights[:, 2] * u_en_test +
                 pred_weights[:, 3] * u_gcn_test)

    results["G-PLS + RF + EN + GCN (Dynamic Gating)"].append(spearmanr(t_gpls_test, u_dynamic)[0])

# Convert to DataFrame
df_results = pd.DataFrame(results)
print("\nMean out-of-sample Spearman correlations:")
print(df_results.mean())
print("\nStandard deviation:")
print(df_results.std())

# Print average dynamic gating weights across CV splits
mean_cortex = np.mean(cortex_weights_accum, axis=0)
mean_subcortex = np.mean(subcortex_weights_accum, axis=0)
print("\nAverage dynamic gating weights across CV splits:")
print(f"Cortex:    G-PLS = {mean_cortex[0]:.4f} | RF = {mean_cortex[1]:.4f} | EN = {mean_cortex[2]:.4f} | GCN = {mean_cortex[3]:.4f}")
print(f"Subcortex: G-PLS = {mean_subcortex[0]:.4f} | RF = {mean_subcortex[1]:.4f} | EN = {mean_subcortex[2]:.4f} | GCN = {mean_subcortex[3]:.4f}")

# Plot results
plt.figure(figsize=(9, 6), dpi=200)
palette_colors = sns.color_palette("Set2", 4)
sns.boxplot(data=df_results, palette=palette_colors, width=0.4, linewidth=1.0, showfliers=False)
sns.stripplot(data=df_results, palette=palette_colors, size=3.0, jitter=True, alpha=0.5, edgecolor="gray", linewidth=0.5)
plt.ylabel("Test score correlation (Spearman)")
plt.xlabel("Model")
plt.title(f"Dynamic Gating vs. Baselines and Static Blend ({n_splits} splits)")
sns.despine(trim=True)

plt.savefig('figs/gpls_dynamic_gating_cv.pdf', bbox_inches='tight')
plt.close()
print("\nDynamic evaluation complete. Plot saved to figs/gpls_dynamic_gating_cv.pdf")
