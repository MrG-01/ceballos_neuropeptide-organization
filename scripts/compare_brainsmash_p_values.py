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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet
from pyls import behavioral_pls
from utils import get_centroids
import re

# Ensure outputs directories exist
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
nulls = np.load('data/receptor_spatial_nulls_Schaefer400_TianS4_HTH.npy')

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

# Family names
family_names = family_genes.columns.tolist()
num_families = len(family_names)

X_fam = zscore(family_genes.values, ddof=1)
Y = zscore(ns.values, ddof=1)

img = nib.load('data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz')
labels = np.unique(img.get_fdata())[1:]
centroids = get_centroids(img, labels=labels)
distance = squareform(pdist(centroids))
N = len(centroids)

# Load FC and SC and pad for Hypothalamus
D_hth = distance[454, :]
sc_raw = np.load('data/template_parc-Schaefer400_TianS4_desc-SC.npy')
sc_padded = np.zeros((N, N))
sc_padded[:454, :454] = sc_raw
sc_hth = np.exp(- (D_hth ** 2) / (2 * 15.0 ** 2))
sc_hth[454] = 0.0
sc_padded[454, :] = sc_hth
sc_padded[:, 454] = sc_hth

fc_raw = np.load('data/template_parc-Schaefer400_TianS4_desc-FC.npy')
fc_padded = np.zeros((N, N))
fc_padded[:454, :454] = np.abs(fc_raw)
closest_subcortex_idx = np.argsort(D_hth)[:4]
closest_subcortex_idx = closest_subcortex_idx[closest_subcortex_idx < 454][:3]
fc_padded[454, :454] = np.abs(fc_raw[closest_subcortex_idx, :]).mean(axis=0)
fc_padded[:454, 454] = fc_padded[454, :454]
fc_padded[454, 454] = 0.0

sc_strength = sc_padded.sum(axis=1)
fc_strength = fc_padded.sum(axis=1)

lut_matched = lut.iloc[:N]
is_cortex = (lut_matched['structure'] == 'cortex').astype(float).values
radial_dist = np.sqrt((centroids ** 2).sum(axis=1))
feats_network = np.column_stack([centroids, radial_dist, is_cortex, sc_strength, fc_strength])

def normalize_adj(A):
    A_tilde = A + np.eye(A.shape[0])
    deg = A_tilde.sum(axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt) | np.isnan(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt

adj_fc = torch.tensor(normalize_adj(fc_padded), dtype=torch.float32)
X_fam_tensor = torch.tensor(X_fam, dtype=torch.float32)

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
#                         FIT EMPIRICAL MODELS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Fitting empirical models...")

# 1. PLSC
plsc = behavioral_pls(X_fam, Y, n_boot=0, n_perm=0, test_split=0)
w_plsc = plsc["x_weights"][:, 0]
u_plsc = Y @ plsc["y_weights"][:, 0]

# 2. G-PLS
gpls = GraphRegularizedPLS(lam=1.0).fit(X_fam, Y, distance)
w_gpls = gpls.x_weights_[:, 0]
u_gpls = Y @ gpls.y_weights_[:, 0]

# 3. Ensemble components
rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1).fit(X_fam, u_gpls)
en = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000).fit(X_fam, u_gpls)

y_target_tensor = torch.tensor(u_gpls.reshape(-1, 1), dtype=torch.float32)
model_gcn = BrainGCN(in_dim=family_genes.shape[1], hidden_dim=64, out_dim=1, dropout=0.5)
opt = torch.optim.Adam(model_gcn.parameters(), lr=0.01, weight_decay=1e-4)

model_gcn.train()
for epoch in range(200):
    opt.zero_grad()
    loss = F.mse_loss(model_gcn(X_fam_tensor, adj_fc), y_target_tensor)
    loss.backward()
    opt.step()

model_gcn.eval()
with torch.no_grad():
    u_gcn = model_gcn(X_fam_tensor, adj_fc).numpy().flatten()
if pearsonr(u_gcn, u_gpls)[0] < 0:
    u_gcn = -u_gcn

# Gating RandomForestClassifier
err_gpls = np.abs(u_gpls - (X_fam @ w_gpls))
err_rf = np.abs(u_gpls - rf.predict(X_fam))
err_en = np.abs(u_gpls - en.predict(X_fam))
err_gcn = np.abs(u_gpls - u_gcn)
errors = np.column_stack([err_gpls, err_rf, err_en, err_gcn])
best_cls = np.argmin(errors, axis=1)

gating = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=35, random_state=42, n_jobs=-1)
gating.fit(feats_network, best_cls)
pred_weights_raw = gating.predict_proba(feats_network)
pred_weights = np.zeros((N, 4))
for idx, cls in enumerate(gating.classes_):
    pred_weights[:, cls] = pred_weights_raw[:, idx]

# Helper to run ensemble prediction
def ensemble_predict(X_in):
    # GPLS
    u_gpls_pred = X_in @ w_gpls
    # RF
    u_rf_pred = rf.predict(X_in)
    if pearsonr(rf.predict(X_fam), u_gpls)[0] < 0: u_rf_pred = -u_rf_pred
    # EN
    u_en_pred = en.predict(X_in)
    if pearsonr(en.predict(X_fam), u_gpls)[0] < 0: u_en_pred = -u_en_pred
    # GCN
    X_in_tensor = torch.tensor(X_in, dtype=torch.float32)
    with torch.no_grad():
        u_gcn_pred = model_gcn(X_in_tensor, adj_fc).numpy().flatten()
    if pearsonr(u_gcn, u_gpls)[0] < 0: u_gcn_pred = -u_gcn_pred
    
    # Gated combination
    u_dynamic_pred = (pred_weights[:, 0] * u_gpls_pred +
                      pred_weights[:, 1] * u_rf_pred +
                      pred_weights[:, 2] * u_en_pred +
                      pred_weights[:, 3] * u_gcn_pred)
    return u_dynamic_pred

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         COMPUTE EMPIRICAL CORRELATIONS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Computing empirical correlations per family...")
emp_corrs_plsc = []
emp_corrs_gpls = []
emp_corrs_ens = []

for f in range(num_families):
    # Isolate family features
    X_rec = np.zeros_like(X_fam)
    X_rec[:, f] = X_fam[:, f]
    
    # Predict
    t_plsc = X_rec @ w_plsc
    t_gpls = X_rec @ w_gpls
    t_ens = ensemble_predict(X_rec)
    
    # Spearman corr with their target scores
    emp_corrs_plsc.append(spearmanr(t_plsc, u_plsc)[0])
    emp_corrs_gpls.append(spearmanr(t_gpls, u_gpls)[0])
    emp_corrs_ens.append(spearmanr(t_ens, u_gpls)[0])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         COMPUTE NULL CORRELATIONS (500 PERMS)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_perms = 500
print(f"Running {n_perms} BrainSMASH permutations per family...")

# Initialize lists of arrays to store null correlations for each family
null_corrs_plsc = [[] for _ in range(num_families)]
null_corrs_gpls = [[] for _ in range(num_families)]
null_corrs_ens = [[] for _ in range(num_families)]

for p in range(n_perms):
    if (p+1) % 100 == 0:
        print(f"Processed {p+1}/{n_perms} permutations...")
        
    # Construct z-scored null family matrix
    X_gene_null = np.zeros((455, 38))
    for j in range(38):
        X_gene_null[:, j] = nulls[j, :, p]
    X_fam_null_df = pd.DataFrame(X_gene_null, columns=receptor_genes.columns)
    X_fam_null_df.columns = [gene_to_family.get(c, c) for c in X_fam_null_df.columns]
    X_fam_null = zscore(X_fam_null_df.groupby(by=X_fam_null_df.columns, axis=1).mean().values, ddof=1)
    
    # Run predictions on isolated null family columns
    for f in range(num_families):
        X_rec_null = np.zeros_like(X_fam_null)
        X_rec_null[:, f] = X_fam_null[:, f]
        
        # Predict
        t_plsc_null = X_rec_null @ w_plsc
        t_gpls_null = X_rec_null @ w_gpls
        t_ens_null = ensemble_predict(X_rec_null)
        
        # Corr
        null_corrs_plsc[f].append(spearmanr(t_plsc_null, u_plsc)[0])
        null_corrs_gpls[f].append(spearmanr(t_gpls_null, u_gpls)[0])
        null_corrs_ens[f].append(spearmanr(t_ens_null, u_gpls)[0])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         COMPUTE P-VALUES
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Computing p-values...")
p_plsc_all = []
p_gpls_all = []
p_ens_all = []

for f in range(num_families):
    # Two-tailed p-values based on spatial nulls
    p_plsc = (np.sum(np.abs(null_corrs_plsc[f]) >= np.abs(emp_corrs_plsc[f])) + 1) / (n_perms + 1)
    p_gpls = (np.sum(np.abs(null_corrs_gpls[f]) >= np.abs(emp_corrs_gpls[f])) + 1) / (n_perms + 1)
    p_ens = (np.sum(np.abs(null_corrs_ens[f]) >= np.abs(emp_corrs_ens[f])) + 1) / (n_perms + 1)
    
    p_plsc_all.append(p_plsc)
    p_gpls_all.append(p_gpls)
    p_ens_all.append(p_ens)

results_df = pd.DataFrame({
    'Family': family_names,
    'PLSC_Empirical_r': emp_corrs_plsc,
    'PLSC_p_value': p_plsc_all,
    'GPLS_Empirical_r': emp_corrs_gpls,
    'GPLS_p_value': p_gpls_all,
    'Ensemble_Empirical_r': emp_corrs_ens,
    'Ensemble_p_value': p_ens_all
})

results_df.to_csv('results/brainsmash_p_values_comparison.csv', index=False)
print("P-value comparison results saved to results/brainsmash_p_values_comparison.csv")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOTTING SCATTER PLOTS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Creating scatter plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300)

# 1. PLSC vs Ensemble
ax1 = axes[0]
sns.scatterplot(data=results_df, x='PLSC_p_value', y='Ensemble_p_value', s=100, color='#4C72B0', edgecolor='k', alpha=0.8, ax=ax1)
ax1.plot([0, 1], [0, 1], color='r', linestyle='--', linewidth=1.5, label='y = x')
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.05, 1.05)
ax1.set_xlabel('PLSC Spatial p-value (BrainSMASH)', fontsize=12)
ax1.set_ylabel('Ensemble Spatial p-value (BrainSMASH)', fontsize=12)
ax1.set_title('PLSC vs Ensemble BrainSMASH P-values', fontsize=14, fontweight='bold', pad=10)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower right')

# Annotate points with family names
for idx, row in results_df.iterrows():
    ax1.annotate(row['Family'], (row['PLSC_p_value'], row['Ensemble_p_value']), 
                 textcoords="offset points", xytext=(0, 5), ha='center', fontsize=7, alpha=0.8)

# 2. G-PLS vs Ensemble
ax2 = axes[1]
sns.scatterplot(data=results_df, x='GPLS_p_value', y='Ensemble_p_value', s=100, color='#55A868', edgecolor='k', alpha=0.8, ax=ax2)
ax2.plot([0, 1], [0, 1], color='r', linestyle='--', linewidth=1.5, label='y = x')
ax2.set_xlim(-0.05, 1.05)
ax2.set_ylim(-0.05, 1.05)
ax2.set_xlabel('G-PLS Spatial p-value (BrainSMASH)', fontsize=12)
ax2.set_ylabel('Ensemble Spatial p-value (BrainSMASH)', fontsize=12)
ax2.set_title('G-PLS vs Ensemble BrainSMASH P-values', fontsize=14, fontweight='bold', pad=10)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='lower right')

# Annotate points with family names
for idx, row in results_df.iterrows():
    ax2.annotate(row['Family'], (row['GPLS_p_value'], row['Ensemble_p_value']), 
                 textcoords="offset points", xytext=(0, 5), ha='center', fontsize=7, alpha=0.8)

plt.tight_layout()
fig.savefig('figs/brainsmash_p_values_comparison.pdf', bbox_inches='tight', dpi=300)
fig.savefig('figs/brainsmash_p_values_comparison.png', bbox_inches='tight', dpi=300)
plt.close(fig)

print("Scatter plot comparisons complete. Plots saved to figs/brainsmash_p_values_comparison.*")
