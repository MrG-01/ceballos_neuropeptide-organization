import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import scipy.stats as sstats
import matplotlib.pyplot as plt
import seaborn as sns
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
from plot_utils import divergent_green_orange

# Ensure directories exist
os.makedirs('figs', exist_ok=True)
os.makedirs('results', exist_ok=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                              LOAD DATA
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Loading data...")
ns = pd.read_csv('./data/neurosynth_Schaefer400_TianS4.csv', index_col=0)
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
lut = pd.read_csv('data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv', index_col=0)
overview = pd.read_csv('data/receptor_overview.csv')

peptide_names = receptor_genes.columns.values
X = zscore(receptor_genes.values, ddof=1)
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
#                         FIT MODELS ON FULL DATA
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Fitting baseline and base models...")
# 1. PLSC
pls_full = behavioral_pls(Y, X, n_boot=0, n_perm=0, test_split=0)
t_plsc = X @ pls_full["y_weights"][:, 0]

# 2. G-PLS
gpls = GraphRegularizedPLS(lam=1.0).fit(X, Y, distance)
t_gpls, u_gpls = gpls.transform(X, Y)
t_gpls, u_gpls = t_gpls[:, 0], u_gpls[:, 0]

# 3. Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1).fit(X, u_gpls)
u_rf = rf.predict(X)
if pearsonr(u_rf, u_gpls)[0] < 0: u_rf = -u_rf

# 4. ElasticNet
en = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000).fit(X, u_gpls)
u_en = en.predict(X)
if pearsonr(u_en, u_gpls)[0] < 0: u_en = -u_en

# 5. BrainGCN
y_target_tensor = torch.tensor(u_gpls.reshape(-1, 1), dtype=torch.float32)
model_gcn = BrainGCN(in_dim=receptor_genes.shape[1], hidden_dim=64, out_dim=1, dropout=0.5)
opt = torch.optim.Adam(model_gcn.parameters(), lr=0.01, weight_decay=1e-4)

model_gcn.train()
for epoch in range(200):
    opt.zero_grad()
    loss = F.mse_loss(model_gcn(X_tensor, adj_fc), y_target_tensor)
    loss.backward()
    opt.step()

model_gcn.eval()
with torch.no_grad():
    u_gcn = model_gcn(X_tensor, adj_fc).numpy().flatten()
if pearsonr(u_gcn, u_gpls)[0] < 0: u_gcn = -u_gcn

# 6. Ensemble Model (Dynamic Gating)
err_gpls = np.abs(u_gpls - t_gpls)
err_rf = np.abs(u_gpls - u_rf)
err_en = np.abs(u_gpls - u_en)
err_gcn = np.abs(u_gpls - u_gcn)
errors = np.column_stack([err_gpls, err_rf, err_en, err_gcn])
best_cls = np.argmin(errors, axis=1)

gating = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=35, random_state=42, n_jobs=-1)
gating.fit(feats_network, best_cls)
pred_weights_raw = gating.predict_proba(feats_network)
pred_weights = np.zeros((N, 4))
for idx, cls in enumerate(gating.classes_):
    pred_weights[:, cls] = pred_weights_raw[:, idx]

u_dynamic = (pred_weights[:, 0] * t_gpls +
             pred_weights[:, 1] * u_rf +
             pred_weights[:, 2] * u_en +
             pred_weights[:, 3] * u_gcn)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         COMPUTE LOADINGS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Computing receptor loadings across models...")
loadings = {}

# Compute Spearman correlation of each gene expression map with the respective predicted score map
loadings["PLSC"] = [spearmanr(X[:, j], t_plsc)[0] for j in range(X.shape[1])]
loadings["G-PLS"] = [spearmanr(X[:, j], t_gpls)[0] for j in range(X.shape[1])]
loadings["Random Forest"] = [spearmanr(X[:, j], u_rf)[0] for j in range(X.shape[1])]
loadings["ElasticNet"] = [spearmanr(X[:, j], u_en)[0] for j in range(X.shape[1])]
loadings["BrainGCN"] = [spearmanr(X[:, j], u_gcn)[0] for j in range(X.shape[1])]
loadings["Ensemble"] = [spearmanr(X[:, j], u_dynamic)[0] for j in range(X.shape[1])]

df_loadings = pd.DataFrame(loadings, index=peptide_names)

# Sign align columns to PLSC for visual consistency
for col in df_loadings.columns:
    if col != "PLSC":
        corr = np.corrcoef(df_loadings["PLSC"], df_loadings[col])[0, 1]
        if corr < 0:
            df_loadings[col] = -df_loadings[col]

# Save loadings comparison CSV
df_loadings.to_csv('results/ensemble_loadings_comparison.csv')
print("Loadings successfully calculated and saved to results/ensemble_loadings_comparison.csv")

# Sort neuropeptides by the absolute ensemble loading values to group strong contributors
sort_idx = df_loadings["Ensemble"].abs().sort_values(ascending=False).index
df_loadings_sorted = df_loadings.loc[sort_idx]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOT LOADINGS COMPARISON HEATMAP
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Generating figures...")
fig, ax = plt.subplots(figsize=(8, 12), dpi=300)

sns.heatmap(df_loadings_sorted, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax,
            cbar_kws={'label': 'Receptor Loading (Spearman $r$)', 'shrink': 0.5},
            linewidths=0.5, linecolor='white')

ax.set_title("Neuropeptide Receptor Loadings Across Ensemble Components", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Model / Component", fontsize=12)
ax.set_ylabel("Neuropeptide Receptor Gene", fontsize=12)

# Color-code y-tick labels by family color and highlight OPRM1 (purple) and OPRK1 (dodgerblue)
overview_filtered = overview[overview['gene'].isin(peptide_names)].copy()
overview_filtered['family'] = overview_filtered['family'].fillna(overview_filtered['gene'])
overview_filtered['family'] = overview_filtered['family'].replace('None', np.nan)
overview_filtered['family'] = overview_filtered['family'].fillna(overview_filtered['gene'])
gene_to_family = dict(zip(overview_filtered['gene'], overview_filtered['family']))

families_unique = sorted(list(set(overview_filtered['family'])))
family_colors = sns.color_palette('tab20', n_colors=len(families_unique))
family_color_map = {fam: color for fam, color in zip(families_unique, family_colors)}

for tick in ax.get_yticklabels():
    gene = tick.get_text()
    tick.set_fontsize(8.5)
    if gene == 'OPRM1':
        tick.set_color('purple')
        tick.set_weight('bold')
    elif gene == 'OPRK1':
        tick.set_color('dodgerblue')
        tick.set_weight('bold')
    else:
        fam = gene_to_family.get(gene)
        if fam and fam in family_color_map:
            tick.set_color(family_color_map[fam])
            tick.set_weight('bold')

plt.tight_layout()
plt.savefig('figs/ensemble_loadings_comparison.pdf', bbox_inches='tight')
plt.savefig('figs/ensemble_loadings_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure successfully generated!")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         COMPUTE TERM LOADINGS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Computing Neurosynth term loadings across models...")
term_names = ns.columns.values
term_loadings = {}

# Compute Spearman correlation of each Neurosynth term map with the respective predicted score map
term_loadings["PLSC"] = [spearmanr(Y[:, k], t_plsc)[0] for k in range(Y.shape[1])]
term_loadings["G-PLS"] = [spearmanr(Y[:, k], t_gpls)[0] for k in range(Y.shape[1])]
term_loadings["Random Forest"] = [spearmanr(Y[:, k], u_rf)[0] for k in range(Y.shape[1])]
term_loadings["ElasticNet"] = [spearmanr(Y[:, k], u_en)[0] for k in range(Y.shape[1])]
term_loadings["BrainGCN"] = [spearmanr(Y[:, k], u_gcn)[0] for k in range(Y.shape[1])]
term_loadings["Ensemble"] = [spearmanr(Y[:, k], u_dynamic)[0] for k in range(Y.shape[1])]

df_term_loadings = pd.DataFrame(term_loadings, index=term_names)

# Sign align columns to PLSC for visual consistency
for col in df_term_loadings.columns:
    if col != "PLSC":
        corr = np.corrcoef(df_term_loadings["PLSC"], df_term_loadings[col])[0, 1]
        if corr < 0:
            df_term_loadings[col] = -df_term_loadings[col]

# Save term loadings comparison CSV
df_term_loadings.to_csv('results/ensemble_term_loadings_comparison.csv')
print("Term loadings successfully calculated and saved to results/ensemble_term_loadings_comparison.csv")

# Select the top 15 positive and top 15 negative terms based on the Ensemble model
sorted_terms_by_ensemble = df_term_loadings["Ensemble"].sort_values(ascending=False)
top_pos_terms = sorted_terms_by_ensemble.head(15).index
top_neg_terms = sorted_terms_by_ensemble.tail(15).index
selected_terms = list(top_pos_terms) + list(top_neg_terms)

df_term_loadings_selected = df_term_loadings.loc[selected_terms]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOT TERM LOADINGS HEATMAP
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Generating term loadings figure...")
fig, ax = plt.subplots(figsize=(8, 10), dpi=300)

sns.heatmap(df_term_loadings_selected, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax,
            cbar_kws={'label': 'Term Loading (Spearman $r$)', 'shrink': 0.5},
            linewidths=0.5, linecolor='white')

ax.set_title("Neurosynth Term Loadings Across Ensemble Components\n(Top 15 Positive and Top 15 Negative Terms)", fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel("Model / Component", fontsize=11)
ax.set_ylabel("Neurosynth Cognitive Term", fontsize=11)

plt.tight_layout()
plt.savefig('figs/ensemble_term_loadings_comparison.pdf', bbox_inches='tight')
plt.savefig('figs/ensemble_term_loadings_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Term loadings figure successfully generated!")

