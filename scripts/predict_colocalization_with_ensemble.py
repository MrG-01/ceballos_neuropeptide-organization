import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import scipy.stats as sstats
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import zscore, spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet
from netneurotools.stats import get_dominance_stats
from plot_utils import divergent_green_orange
from utils import get_centroids

# Ensure figs and results directories exist
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
nt_densities = pd.read_csv('data/annotations/nt_receptor_densities_Schaefer400_TianS4_HTH.csv', index_col=0)

peptide_names = receptor_genes.columns.values
nt_names = nt_densities.columns.values

# Standardize inputs
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
print("Fitting base models and gating classifier on full dataset...")
gpls = GraphRegularizedPLS(lam=1.0).fit(X, Y, distance)
t_gpls, u_gpls = gpls.transform(X, Y)
t_gpls, u_gpls = t_gpls[:, 0], u_gpls[:, 0]

rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1).fit(X, u_gpls)
u_rf = rf.predict(X)
if pearsonr(u_rf, u_gpls)[0] < 0: u_rf = -u_rf

en = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000).fit(X, u_gpls)
u_en = en.predict(X)
if pearsonr(u_en, u_gpls)[0] < 0: u_en = -u_en

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

# Gating RandomForestClassifier
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         GENERATE GENE PREDICTIONS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Generating ensemble predictions for individual neuropeptide genes...")
predicted_maps = {}
for idx, gene_name in enumerate(peptide_names):
    X_rec = np.zeros_like(X)
    X_rec[:, idx] = X[:, idx]
    X_rec_tensor = torch.tensor(X_rec, dtype=torch.float32)
    
    u_gpls_rec = X_rec @ gpls.x_weights_[:, 0]
    
    u_rf_rec = rf.predict(X_rec)
    if pearsonr(rf.predict(X), u_gpls)[0] < 0: u_rf_rec = -u_rf_rec
    
    u_en_rec = en.predict(X_rec)
    if pearsonr(en.predict(X), u_gpls)[0] < 0: u_en_rec = -u_en_rec
    
    with torch.no_grad():
        u_gcn_rec = model_gcn(X_rec_tensor, adj_fc).numpy().flatten()
    if pearsonr(u_gcn, u_gpls)[0] < 0: u_gcn_rec = -u_gcn_rec
    
    u_dynamic_rec = (pred_weights[:, 0] * u_gpls_rec +
                     pred_weights[:, 1] * u_rf_rec +
                     pred_weights[:, 2] * u_en_rec +
                     pred_weights[:, 3] * u_gcn_rec)
    
    u_min, u_max = u_dynamic_rec.min(), u_dynamic_rec.max()
    if u_max > u_min:
        u_norm = (u_dynamic_rec - u_min) / (u_max - u_min)
    else:
        u_norm = np.zeros_like(u_dynamic_rec)
        
    predicted_maps[gene_name] = u_norm

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         DOMINANCE ANALYSIS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Performing dominance analysis with predicted maps...")
da_fn = 'results/da_nt_peptides_ensemble_total_dominance.npy'
if os.path.exists(da_fn):
    dom_total = np.load(da_fn)
else:
    dom_list = []
    # PET densities standardized as predictors
    X_pet = zscore(nt_densities.values, ddof=1)
    
    for name in peptide_names:
        y_pred = zscore(predicted_maps[name][:-1], ddof=1)
        model_metrics, model_r_sq = get_dominance_stats(X_pet, y_pred, n_jobs=-1)
        dom_list.append((model_metrics, model_r_sq))
    dom_total = [_[0]["total_dominance"] for _ in dom_list]
    dom_total = np.array(dom_total)
    np.save(da_fn, dom_total)

dom_rel = (dom_total / dom_total.sum(axis=0)) * 100

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOT DOMINANCE HEATMAP
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Generating figures...")
df = pd.DataFrame(dom_total.T, columns=peptide_names, index=nt_names)
receptors_by_dominance = df.sum(axis=0).sort_values(ascending=True).index
pep_idx = [df.columns.get_loc(_) for _ in receptors_by_dominance]
df = df[receptors_by_dominance]

# Group by Metab/Iono
nt_classes = pd.read_csv('data/annotations/nt_receptor_classes.csv', index_col=0)
mi = nt_classes['Metab/Iono'].loc[nt_names]
idf = df.loc[mi[mi == 'ionotropic'].index]
mdf = df.loc[mi[mi == 'metabotropic'].index]
df = pd.concat((idf, mdf), axis=0)
nt_idx = [np.where(nt_names == _)[0][0] for _ in df.index]

plot_df = pd.DataFrame(dom_rel[pep_idx], index=receptors_by_dominance, columns=nt_names)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 10), gridspec_kw={'width_ratios': [1, 3]}, dpi=300)

palette = divergent_green_orange(n_colors=9, return_palette=True)
orange_color = [palette[1], palette[-2]][1]

sns.barplot(x=df.sum(axis=0), y=receptors_by_dominance, ax=ax1, color=orange_color)
ax1.set_xlabel('R$^2$', fontsize=14)
ax1.set_xlim(0, 1)
ax1.set_yticks([])
ax1.invert_xaxis()
sns.despine(ax=ax1, left=True)

max_val = plot_df.max().round(0).max()
sns.heatmap(plot_df, cbar_kws={'shrink': 0.5}, ax=ax2, cmap=divergent_green_orange(), 
            center=0, vmin=0, vmax=max_val, linecolor='white', linewidths=0.5)

# Color y-tick labels by family color and highlight OPRM1 (purple) and OPRK1 (dodgerblue)
overview_filtered = overview[overview['gene'].isin(peptide_names)].copy()
overview_filtered['family'] = overview_filtered['family'].fillna(overview_filtered['gene'])
overview_filtered['family'] = overview_filtered['family'].replace('None', np.nan)
overview_filtered['family'] = overview_filtered['family'].fillna(overview_filtered['gene'])
gene_to_family = dict(zip(overview_filtered['gene'], overview_filtered['family']))

families_unique = sorted(list(set(overview_filtered['family'])))
family_colors = sns.color_palette('tab20', n_colors=len(families_unique))
family_color_map = {fam: color for fam, color in zip(families_unique, family_colors)}

for tick in ax2.get_yticklabels():
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

cbar = ax2.collections[0].colorbar
cbar.set_label('Relative contribution (%)', size=14)
ax2.set_xticklabels(plot_df.columns, rotation=90, horizontalalignment='center')
plt.tight_layout()

plt.savefig('figs/colocalization_nt_peptides_ensemble.pdf')
plt.savefig('figs/colocalization_nt_peptides_ensemble.png', dpi=300)
plt.close()
print("Figure successfully generated!")
