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
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from pyls import behavioral_pls
from utils import get_centroids

# Ensure output directories exist
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

auc_results = {
    "Split": [],
    # ROC-AUC
    "PLSC_AUC_Median": [], "GPLS_AUC_Median": [], "Ensemble_AUC_Median": [],
    "PLSC_AUC_80th": [], "GPLS_AUC_80th": [], "Ensemble_AUC_80th": [],
    # PR-AUC / Average Precision (AP)
    "PLSC_AP_Median": [], "GPLS_AP_Median": [], "Ensemble_AP_Median": [],
    "PLSC_AP_80th": [], "GPLS_AP_80th": [], "Ensemble_AP_80th": []
}

# Grids for interpolations
mean_fpr = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 100)

# True Positive Rates for averaging ROC curves
tprs_plsc_med, tprs_gpls_med, tprs_ens_med = [], [], []
tprs_plsc_80, tprs_gpls_80, tprs_ens_80 = [], [], []

# Precision values for averaging PR curves
precs_plsc_med, precs_gpls_med, precs_ens_med = [], [], []
precs_plsc_80, precs_gpls_80, precs_ens_80 = [], [], []

print(f"Running {n_splits}-fold cross-validation for ROC-AUC and PR-AUC comparison...")

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
    except Exception as e:
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

    # Errors on training set
    err_gpls = np.abs(u_gpls_train - t_gpls_train)
    err_rf = np.abs(u_gpls_train - u_rf_train)
    err_en = np.abs(u_gpls_train - u_en_train)
    err_gcn = np.abs(u_gpls_train - u_gcn_train)

    # Gating Classifier training
    best_model_idx = np.argmin(np.column_stack([err_gpls, err_rf, err_en, err_gcn]), axis=1)
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

    # Gated combination
    u_dynamic = (pred_weights[:, 0] * u_gpls_test + 
                 pred_weights[:, 1] * u_rf_test + 
                 pred_weights[:, 2] * u_en_test +
                 pred_weights[:, 3] * u_gcn_test)

    # Align ensemble sign
    if pearsonr(u_dynamic, u_gpls_test)[0] < 0:
        u_dynamic = -u_dynamic

    # Binarize Ground Truth Targets
    # A. Median Split
    y_plsc_med = (u_plsc_test > np.median(u_plsc_test)).astype(int)
    y_gpls_med = (u_gpls_test > np.median(u_gpls_test)).astype(int)
    y_ens_med = (u_gpls_test > np.median(u_gpls_test)).astype(int)

    # B. 80th Percentile Split (Top 20% Peak Activation Hubs)
    y_plsc_80 = (u_plsc_test > np.percentile(u_plsc_test, 80)).astype(int)
    y_gpls_80 = (u_gpls_test > np.percentile(u_gpls_test, 80)).astype(int)
    y_ens_80 = (u_gpls_test > np.percentile(u_gpls_test, 80)).astype(int)

    # 1. Compute ROC-AUC scores
    auc_plsc_med = roc_auc_score(y_plsc_med, t_plsc_test)
    auc_gpls_med = roc_auc_score(y_gpls_med, t_gpls_test)
    auc_ens_med = roc_auc_score(y_ens_med, u_dynamic)

    auc_plsc_80 = roc_auc_score(y_plsc_80, t_plsc_test)
    auc_gpls_80 = roc_auc_score(y_gpls_80, t_gpls_test)
    auc_ens_80 = roc_auc_score(y_ens_80, u_dynamic)

    # 2. Compute Average Precision (PR-AUC) scores
    ap_plsc_med = average_precision_score(y_plsc_med, t_plsc_test)
    ap_gpls_med = average_precision_score(y_gpls_med, t_gpls_test)
    ap_ens_med = average_precision_score(y_ens_med, u_dynamic)

    ap_plsc_80 = average_precision_score(y_plsc_80, t_plsc_test)
    ap_gpls_80 = average_precision_score(y_gpls_80, t_gpls_test)
    ap_ens_80 = average_precision_score(y_ens_80, u_dynamic)

    # ROC curve interpolations
    # 1. Median Threshold
    fpr, tpr, _ = roc_curve(y_plsc_med, t_plsc_test)
    tprs_plsc_med.append(np.interp(mean_fpr, fpr, tpr))
    tprs_plsc_med[-1][0] = 0.0

    fpr, tpr, _ = roc_curve(y_gpls_med, t_gpls_test)
    tprs_gpls_med.append(np.interp(mean_fpr, fpr, tpr))
    tprs_gpls_med[-1][0] = 0.0

    fpr, tpr, _ = roc_curve(y_ens_med, u_dynamic)
    tprs_ens_med.append(np.interp(mean_fpr, fpr, tpr))
    tprs_ens_med[-1][0] = 0.0

    # 2. 80th Percentile Threshold
    fpr, tpr, _ = roc_curve(y_plsc_80, t_plsc_test)
    tprs_plsc_80.append(np.interp(mean_fpr, fpr, tpr))
    tprs_plsc_80[-1][0] = 0.0

    fpr, tpr, _ = roc_curve(y_gpls_80, t_gpls_test)
    tprs_gpls_80.append(np.interp(mean_fpr, fpr, tpr))
    tprs_gpls_80[-1][0] = 0.0

    fpr, tpr, _ = roc_curve(y_ens_80, u_dynamic)
    tprs_ens_80.append(np.interp(mean_fpr, fpr, tpr))
    tprs_ens_80[-1][0] = 0.0

    # Precision-Recall curve interpolations (Reverse recall to be sorted ascendingly for np.interp)
    # 1. Median Threshold
    p_curve, r_curve, _ = precision_recall_curve(y_plsc_med, t_plsc_test)
    precs_plsc_med.append(np.interp(mean_recall, r_curve[::-1], p_curve[::-1]))

    p_curve, r_curve, _ = precision_recall_curve(y_gpls_med, t_gpls_test)
    precs_gpls_med.append(np.interp(mean_recall, r_curve[::-1], p_curve[::-1]))

    p_curve, r_curve, _ = precision_recall_curve(y_ens_med, u_dynamic)
    precs_ens_med.append(np.interp(mean_recall, r_curve[::-1], p_curve[::-1]))

    # 2. 80th Percentile Threshold
    p_curve, r_curve, _ = precision_recall_curve(y_plsc_80, t_plsc_test)
    precs_plsc_80.append(np.interp(mean_recall, r_curve[::-1], p_curve[::-1]))

    p_curve, r_curve, _ = precision_recall_curve(y_gpls_80, t_gpls_test)
    precs_gpls_80.append(np.interp(mean_recall, r_curve[::-1], p_curve[::-1]))

    p_curve, r_curve, _ = precision_recall_curve(y_ens_80, u_dynamic)
    precs_ens_80.append(np.interp(mean_recall, r_curve[::-1], p_curve[::-1]))

    # Store results
    auc_results["Split"].append(i)
    auc_results["PLSC_AUC_Median"].append(auc_plsc_med)
    auc_results["GPLS_AUC_Median"].append(auc_gpls_med)
    auc_results["Ensemble_AUC_Median"].append(auc_ens_med)
    auc_results["PLSC_AUC_80th"].append(auc_plsc_80)
    auc_results["GPLS_AUC_80th"].append(auc_gpls_80)
    auc_results["Ensemble_AUC_80th"].append(auc_ens_80)
    
    auc_results["PLSC_AP_Median"].append(ap_plsc_med)
    auc_results["GPLS_AP_Median"].append(ap_gpls_med)
    auc_results["Ensemble_AP_Median"].append(ap_ens_med)
    auc_results["PLSC_AP_80th"].append(ap_plsc_80)
    auc_results["GPLS_AP_80th"].append(ap_gpls_80)
    auc_results["Ensemble_AP_80th"].append(ap_ens_80)

df_auc = pd.DataFrame(auc_results)
df_auc.to_csv('results/ensemble_auc_comparison.csv', index=False)
print("Classification ROC-AUC and PR-AUC scores saved to results/ensemble_auc_comparison.csv")

# Print Summary Statistics
print("\n" + "="*85)
print("                      OUT-OF-SAMPLE CLASSIFICATION PERFORMANCE SUMMARY")
print("="*85)
print(f"Model      | Median Split AUC | 80th Perc Split AUC | Median Split AP  | 80th Perc Split AP")
print("-"*85)
print(f"PLSC       | {df_auc['PLSC_AUC_Median'].mean():.4f} ± {df_auc['PLSC_AUC_Median'].std():.3f} | {df_auc['PLSC_AUC_80th'].mean():.4f} ± {df_auc['PLSC_AUC_80th'].std():.3f}  | {df_auc['PLSC_AP_Median'].mean():.4f} ± {df_auc['PLSC_AP_Median'].std():.3f} | {df_auc['PLSC_AP_80th'].mean():.4f} ± {df_auc['PLSC_AP_80th'].std():.3f}")
print(f"G-PLS      | {df_auc['GPLS_AUC_Median'].mean():.4f} ± {df_auc['GPLS_AUC_Median'].std():.3f} | {df_auc['GPLS_AUC_80th'].mean():.4f} ± {df_auc['GPLS_AUC_80th'].std():.3f}  | {df_auc['GPLS_AP_Median'].mean():.4f} ± {df_auc['GPLS_AP_Median'].std():.3f} | {df_auc['GPLS_AP_80th'].mean():.4f} ± {df_auc['GPLS_AP_80th'].std():.3f}")
print(f"Ensemble   | {df_auc['Ensemble_AUC_Median'].mean():.4f} ± {df_auc['Ensemble_AUC_Median'].std():.3f} | {df_auc['Ensemble_AUC_80th'].mean():.4f} ± {df_auc['Ensemble_AUC_80th'].std():.3f}  | {df_auc['Ensemble_AP_Median'].mean():.4f} ± {df_auc['Ensemble_AP_Median'].std():.3f} | {df_auc['Ensemble_AP_80th'].mean():.4f} ± {df_auc['Ensemble_AP_80th'].std():.3f}")
print("="*85)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOTTING FIGURES
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Generating plots...")

sns.set_theme(style="whitegrid")
palette_colors = ["#4C72B0", "#55A868", "#C44E52"] # PLSC, G-PLS, Ensemble
model_labels = ["PLSC", "G-PLS", "Ensemble"]

# 1. Figure 1: ROC-AUC Boxplots (1x2 Panel)
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)
df_med = df_auc[["PLSC_AUC_Median", "GPLS_AUC_Median", "Ensemble_AUC_Median"]].copy()
df_med.columns = model_labels
sns.boxplot(data=df_med, palette=palette_colors, width=0.4, linewidth=1.2, showfliers=False, ax=axes1[0])
sns.stripplot(data=df_med, palette=palette_colors, size=3.0, jitter=0.2, alpha=0.4, edgecolor="gray", linewidth=0.5, ax=axes1[0])
axes1[0].set_ylabel("Out-of-Sample ROC-AUC Score", fontsize=11, fontweight='bold', labelpad=10)
axes1[0].set_title("Median Split (50th Percentile Target)", fontsize=12, fontweight='bold', pad=15)
axes1[0].set_ylim(0.2, 1.05)

df_80 = df_auc[["PLSC_AUC_80th", "GPLS_AUC_80th", "Ensemble_AUC_80th"]].copy()
df_80.columns = model_labels
sns.boxplot(data=df_80, palette=palette_colors, width=0.4, linewidth=1.2, showfliers=False, ax=axes1[1])
sns.stripplot(data=df_80, palette=palette_colors, size=3.0, jitter=0.2, alpha=0.4, edgecolor="gray", linewidth=0.5, ax=axes1[1])
axes1[1].set_ylabel("Out-of-Sample ROC-AUC Score", fontsize=11, fontweight='bold', labelpad=10)
axes1[1].set_title("High-Intensity Split (80th Percentile Target)", fontsize=12, fontweight='bold', pad=15)
axes1[1].set_ylim(0.2, 1.05)

for ax in axes1:
    sns.despine(ax=ax, trim=True)
plt.tight_layout()
fig1.savefig('figs/ensemble_auc_comparison.pdf', bbox_inches='tight')
fig1.savefig('figs/ensemble_auc_comparison.png', bbox_inches='tight')
plt.close(fig1)

# 2. Figure 2: Mean ROC Curves (1x2 Panel)
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)

for idx, (tprs, label) in enumerate([
    (tprs_plsc_med, "PLSC"), (tprs_gpls_med, "G-PLS"), (tprs_ens_med, "Ensemble")
]):
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    mean_val = df_auc[f"{label.replace('-', '')}_AUC_Median"].mean()
    std_val = df_auc[f"{label.replace('-', '')}_AUC_Median"].std()
    axes2[0].plot(mean_fpr, mean_tpr, color=palette_colors[idx], linewidth=2.0, 
                   label=f"{label} (AUC = {mean_val:.3f} ± {std_val:.3f})")
    axes2[0].fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1), 
                            color=palette_colors[idx], alpha=0.15)

axes2[0].plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.0)
axes2[0].set_xlim([-0.02, 1.02])
axes2[0].set_ylim([-0.02, 1.02])
axes2[0].set_xlabel("False Positive Rate (FPR)", fontsize=11, fontweight='bold')
axes2[0].set_ylabel("True Positive Rate (TPR)", fontsize=11, fontweight='bold')
axes2[0].set_title("ROC Curves: Median Split", fontsize=12, fontweight='bold', pad=15)
axes2[0].legend(loc="lower right")

for idx, (tprs, label) in enumerate([
    (tprs_plsc_80, "PLSC"), (tprs_gpls_80, "G-PLS"), (tprs_ens_80, "Ensemble")
]):
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    mean_val = df_auc[f"{label.replace('-', '')}_AUC_80th"].mean()
    std_val = df_auc[f"{label.replace('-', '')}_AUC_80th"].std()
    axes2[1].plot(mean_fpr, mean_tpr, color=palette_colors[idx], linewidth=2.0, 
                   label=f"{label} (AUC = {mean_val:.3f} ± {std_val:.3f})")
    axes2[1].fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1), 
                            color=palette_colors[idx], alpha=0.15)

axes2[1].plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.0)
axes2[1].set_xlim([-0.02, 1.02])
axes2[1].set_ylim([-0.02, 1.02])
axes2[1].set_xlabel("False Positive Rate (FPR)", fontsize=11, fontweight='bold')
axes2[1].set_ylabel("True Positive Rate (TPR)", fontsize=11, fontweight='bold')
axes2[1].set_title("ROC Curves: 80th Percentile Split", fontsize=12, fontweight='bold', pad=15)
axes2[1].legend(loc="lower right")

for ax in axes2:
    sns.despine(ax=ax, trim=True)
plt.tight_layout()
fig2.savefig('figs/ensemble_roc_curves.pdf', bbox_inches='tight')
fig2.savefig('figs/ensemble_roc_curves.png', bbox_inches='tight')
plt.close(fig2)


# 3. Figure 3: PR-AUC (Average Precision) Boxplots (1x2 Panel)
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)
df_ap_med = df_auc[["PLSC_AP_Median", "GPLS_AP_Median", "Ensemble_AP_Median"]].copy()
df_ap_med.columns = model_labels
sns.boxplot(data=df_ap_med, palette=palette_colors, width=0.4, linewidth=1.2, showfliers=False, ax=axes3[0])
sns.stripplot(data=df_ap_med, palette=palette_colors, size=3.0, jitter=0.2, alpha=0.4, edgecolor="gray", linewidth=0.5, ax=axes3[0])
axes3[0].set_ylabel("Out-of-Sample Average Precision (AP)", fontsize=11, fontweight='bold', labelpad=10)
axes3[0].set_title("Median Split (50% Positives)", fontsize=12, fontweight='bold', pad=15)
axes3[0].set_ylim(0.2, 1.05)

df_ap_80 = df_auc[["PLSC_AP_80th", "GPLS_AP_80th", "Ensemble_AP_80th"]].copy()
df_ap_80.columns = model_labels
sns.boxplot(data=df_ap_80, palette=palette_colors, width=0.4, linewidth=1.2, showfliers=False, ax=axes3[1])
sns.stripplot(data=df_ap_80, palette=palette_colors, size=3.0, jitter=0.2, alpha=0.4, edgecolor="gray", linewidth=0.5, ax=axes3[1])
axes3[1].set_ylabel("Out-of-Sample Average Precision (AP)", fontsize=11, fontweight='bold', labelpad=10)
axes3[1].set_title("High-Intensity Split (20% Positives)", fontsize=12, fontweight='bold', pad=15)
axes3[1].set_ylim(0.0, 1.05)

for ax in axes3:
    sns.despine(ax=ax, trim=True)
plt.tight_layout()
fig3.savefig('figs/ensemble_pr_comparison.pdf', bbox_inches='tight')
fig3.savefig('figs/ensemble_pr_comparison.png', bbox_inches='tight')
plt.close(fig3)


# 4. Figure 4: Average PR Curves (1x2 Panel)
fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)

for idx, (precs, label) in enumerate([
    (precs_plsc_med, "PLSC"), (precs_gpls_med, "G-PLS"), (precs_ens_med, "Ensemble")
]):
    mean_prec = np.mean(precs, axis=0)
    std_prec = np.std(precs, axis=0)
    mean_val = df_auc[f"{label.replace('-', '')}_AP_Median"].mean()
    std_val = df_auc[f"{label.replace('-', '')}_AP_Median"].std()
    axes4[0].plot(mean_recall, mean_prec, color=palette_colors[idx], linewidth=2.0, 
                   label=f"{label} (AP = {mean_val:.3f} ± {std_val:.3f})")
    axes4[0].fill_between(mean_recall, np.maximum(mean_prec - std_prec, 0), np.minimum(mean_prec + std_prec, 1), 
                            color=palette_colors[idx], alpha=0.15)

axes4[0].axhline(y=0.5, linestyle='--', color='gray', linewidth=1.0, label="Random Guess (0.50)")
axes4[0].set_xlim([-0.02, 1.02])
axes4[0].set_ylim([-0.02, 1.02])
axes4[0].set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight='bold')
axes4[0].set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight='bold')
axes4[0].set_title("PR Curves: Median Split", fontsize=12, fontweight='bold', pad=15)
axes4[0].legend(loc="lower left")

for idx, (precs, label) in enumerate([
    (precs_plsc_80, "PLSC"), (precs_gpls_80, "G-PLS"), (precs_ens_80, "Ensemble")
]):
    mean_prec = np.mean(precs, axis=0)
    std_prec = np.std(precs, axis=0)
    mean_val = df_auc[f"{label.replace('-', '')}_AP_80th"].mean()
    std_val = df_auc[f"{label.replace('-', '')}_AP_80th"].std()
    axes4[1].plot(mean_recall, mean_prec, color=palette_colors[idx], linewidth=2.0, 
                   label=f"{label} (AP = {mean_val:.3f} ± {std_val:.3f})")
    axes4[1].fill_between(mean_recall, np.maximum(mean_prec - std_prec, 0), np.minimum(mean_prec + std_prec, 1), 
                            color=palette_colors[idx], alpha=0.15)

axes4[1].axhline(y=0.2, linestyle='--', color='gray', linewidth=1.0, label="Random Guess (0.20)")
axes4[1].set_xlim([-0.02, 1.02])
axes4[1].set_ylim([-0.02, 1.02])
axes4[1].set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight='bold')
axes4[1].set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight='bold')
axes4[1].set_title("PR Curves: 80th Percentile Split", fontsize=12, fontweight='bold', pad=15)
axes4[1].legend(loc="lower left")

for ax in axes4:
    sns.despine(ax=ax, trim=True)
plt.tight_layout()
fig4.savefig('figs/ensemble_pr_curves.pdf', bbox_inches='tight')
fig4.savefig('figs/ensemble_pr_curves.png', bbox_inches='tight')
plt.close(fig4)

print("Classification ROC and PR plots successfully saved!")
