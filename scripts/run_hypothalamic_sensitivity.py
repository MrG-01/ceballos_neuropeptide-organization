import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_centroids

# Ensure figs directory exists
os.makedirs('figs', exist_ok=True)

# Load data
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

family_genes = receptor_genes.copy()
family_genes.columns = [gene_to_family.get(c, c) for c in family_genes.columns]
family_genes = family_genes.groupby(by=family_genes.columns, axis=1).mean()

X_fam = zscore(family_genes.values, ddof=1)
Y = zscore(ns.values, ddof=1)

img = nib.load('data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz')
labels = np.unique(img.get_fdata())[1:]
centroids = get_centroids(img, labels=labels)
distance = squareform(pdist(centroids))
N = len(centroids)

# Load raw matrices (SC & FC)
sc_raw = np.load('data/template_parc-Schaefer400_TianS4_desc-SC.npy')
fc_raw = np.load('data/template_parc-Schaefer400_TianS4_desc-FC.npy')

# Centroids distance to hypothalamus (index 454)
D_hth = distance[454, :]

def get_padded_connectivity(sigma):
    # SC padding
    sc_padded = np.zeros((N, N))
    sc_padded[:454, :454] = sc_raw
    sc_hth = np.exp(- (D_hth ** 2) / (2 * sigma ** 2))
    sc_hth[454] = 0.0
    sc_padded[454, :] = sc_hth
    sc_padded[:, 454] = sc_hth

    # FC padding (using sigma to weigh the subcortical neighbors instead of uniform average)
    fc_padded = np.zeros((N, N))
    fc_padded[:454, :454] = np.abs(fc_raw)
    
    # Weight neighbors based on Gaussian distance kernel
    weights = np.exp(- (D_hth[:454] ** 2) / (2 * sigma ** 2))
    weights /= weights.sum()
    
    fc_padded[454, :454] = (np.abs(fc_raw) * weights[:, np.newaxis]).sum(axis=0)
    fc_padded[:454, 454] = fc_padded[454, :454]
    fc_padded[454, 454] = 0.0
    
    return sc_padded, fc_padded

def normalize_adj(A):
    A_tilde = A + np.eye(A.shape[0])
    deg = A_tilde.sum(axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt) | np.isnan(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt

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

sigmas = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0]
n_splits = 20
test_size = 0.7

sigma_results = {s: [] for s in sigmas}

print("Running Hypothalamic Sensitivity Analysis over 20 splits...")

for s in sigmas:
    print(f"Evaluating sigma = {s}...")
    sc_padded, fc_padded = get_padded_connectivity(s)
    
    # Strengths for gating
    sc_strength = sc_padded.sum(axis=1)
    fc_strength = fc_padded.sum(axis=1)
    
    lut_matched = lut.iloc[:N]
    is_cortex = (lut_matched['structure'] == 'cortex').astype(float).values
    radial_dist = np.sqrt((centroids ** 2).sum(axis=1))
    feats_network = np.column_stack([centroids, radial_dist, is_cortex, sc_strength, fc_strength])
    
    # Normalize FC matrix for GCN
    adj_fc = torch.tensor(normalize_adj(fc_padded), dtype=torch.float32)
    X_fam_tensor = torch.tensor(X_fam, dtype=torch.float32)
    
    for i in range(n_splits):
        train_idx, test_idx = train_test_split(np.arange(X_fam.shape[0]), test_size=test_size, random_state=i)
        Ytrain, Ytest = Y[train_idx], Y[test_idx]
        dist_train = distance[train_idx][:, train_idx]
        
        f_net_train, f_net_test = feats_network[train_idx], feats_network[test_idx]
        
        # Base PLS
        gpls_fam = GraphRegularizedPLS(lam=1.0).fit(X_fam[train_idx], Ytrain, dist_train)
        t_gpls_train, u_gpls_train = gpls_fam.transform(X_fam[train_idx], Ytrain)
        t_gpls_test, u_gpls_test = gpls_fam.transform(X_fam[test_idx], Ytest)
        t_gpls_train, u_gpls_train = t_gpls_train[:, 0], u_gpls_train[:, 0]
        t_gpls_test, u_gpls_test = t_gpls_test[:, 0], u_gpls_test[:, 0]
        
        # Base RF (max_depth=6)
        rf_fam = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=i, n_jobs=-1).fit(X_fam[train_idx], u_gpls_train)
        u_rf_train = rf_fam.predict(X_fam[train_idx])
        u_rf_test = rf_fam.predict(X_fam[test_idx])
        if pearsonr(u_rf_train, u_gpls_train)[0] < 0:
            u_rf_train, u_rf_test = -u_rf_train, -u_rf_test

        # Base EN
        en_fam = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=i, max_iter=2000).fit(X_fam[train_idx], u_gpls_train)
        u_en_train = en_fam.predict(X_fam[train_idx])
        u_en_test = en_fam.predict(X_fam[test_idx])
        if pearsonr(u_en_train, u_gpls_train)[0] < 0:
            u_en_train, u_en_test = -u_en_train, -u_en_test

        # Fit FC GCN
        u_true_train = Ytrain @ gpls_fam.y_weights_[:, 0]
        y_tgt_fam = torch.tensor(np.zeros((N, 1)), dtype=torch.float32)
        y_tgt_fam[train_idx, 0] = torch.tensor(u_true_train, dtype=torch.float32)

        model_gcn_fc = BrainGCN(in_dim=family_genes.shape[1], hidden_dim=64, out_dim=1, dropout=0.5)
        opt_fc = torch.optim.Adam(model_gcn_fc.parameters(), lr=0.01, weight_decay=1e-4)
        model_gcn_fc.train()
        for _ in range(200):
            opt_fc.zero_grad()
            loss = F.mse_loss(model_gcn_fc(X_fam_tensor, adj_fc)[train_idx], y_tgt_fam[train_idx])
            loss.backward()
            opt_fc.step()
        model_gcn_fc.eval()
        with torch.no_grad():
            u_gcn_fc_full = model_gcn_fc(X_fam_tensor, adj_fc).numpy().flatten()
        u_gcn_fc_train = u_gcn_fc_full[train_idx]
        u_gcn_fc_test = u_gcn_fc_full[test_idx]
        if pearsonr(u_gcn_fc_train, u_true_train)[0] < 0:
            u_gcn_fc_train, u_gcn_fc_test = -u_gcn_fc_train, -u_gcn_fc_test

        # Gating RandomForestClassifier (n_estimators=150, max_depth=5, min_samples_leaf=35)
        err_gpls = np.abs(u_true_train - t_gpls_train)
        err_rf = np.abs(u_true_train - u_rf_train)
        err_en = np.abs(u_true_train - u_en_train)
        err_gcn = np.abs(u_true_train - u_gcn_fc_train)
        
        errors = np.column_stack([err_gpls, err_rf, err_en, err_gcn])
        best_cls = np.argmin(errors, axis=1)

        gating = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=35, random_state=i, n_jobs=-1)
        gating.fit(f_net_train, best_cls)
        p_raw = gating.predict_proba(f_net_test)
        p = np.zeros((len(test_idx), 4))
        for c_idx, cls in enumerate(gating.classes_):
            p[:, cls] = p_raw[:, c_idx]
            
        u_dyn = p[:,0]*t_gpls_test + p[:,1]*u_rf_test + p[:,2]*u_en_test + p[:,3]*u_gcn_fc_test
        score = spearmanr(t_gpls_test, u_dyn)[0]
        sigma_results[s].append(score)

# Print results
print("\n--- Sensitivity Results ---")
for s in sigmas:
    scores = sigma_results[s]
    print(f"Sigma = {s:4.1f}mm: mean = {np.mean(scores):.4f}, std = {np.std(scores):.4f}")

# Plotting results
plt.figure(figsize=(7, 5), dpi=300)
df_plot = pd.DataFrame(sigma_results)
sns.boxplot(data=df_plot, palette='viridis_r', width=0.4, linewidth=1.0)
sns.stripplot(data=df_plot, color='black', alpha=0.3, size=3.0, jitter=True)
plt.title("Ensemble Performance Sensitivity to Hypothalamic Padding Width ($\sigma$)", fontweight='bold')
plt.xlabel("Gaussian Kernel Width $\sigma$ (mm)", fontweight='bold')
plt.ylabel("Out-of-sample Spearman Correlation (20 splits)", fontweight='bold')
sns.despine()
plt.savefig('figs/hypothalamus_sensitivity_analysis.pdf', bbox_inches='tight')
plt.close()
print("\nSensitivity analysis complete. Plot saved to figs/hypothalamus_sensitivity_analysis.pdf")
