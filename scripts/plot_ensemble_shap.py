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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet
import shap
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_centroids
from plot_utils import divergent_green_orange
from surfplot import Plot
from brainspace.datasets import load_parcellation
from neuromaps.datasets import fetch_fslr

# Ensure output directories exist
os.makedirs('figs', exist_ok=True)
os.makedirs('results', exist_ok=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         LOAD DATA
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

# Family names
family_names = family_genes.columns.tolist()
vip_family_name = gene_to_family.get('VIPR1', 'Glucagon/ secretin')
print(f"VIPR1 maps to family: {vip_family_name}")
vip_family_idx = family_names.index(vip_family_name)

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
#                         FIT MODELS ON FULL DATA
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Fitting base models and gating classifier on full dataset...")
gpls = GraphRegularizedPLS(lam=1.0).fit(X_fam, Y, distance)
t_gpls, u_gpls = gpls.transform(X_fam, Y)
t_gpls, u_gpls = t_gpls[:, 0], u_gpls[:, 0]

rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1).fit(X_fam, u_gpls)
u_rf = rf.predict(X_fam)
if pearsonr(u_rf, u_gpls)[0] < 0:
    u_rf = -u_rf

en = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000).fit(X_fam, u_gpls)
u_en = en.predict(X_fam)
if pearsonr(u_en, u_gpls)[0] < 0:
    u_en = -u_en

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

# Cache pearson sign checks to avoid repeat computation in loops
rf_sign = -1.0 if pearsonr(rf.predict(X_fam), u_gpls)[0] < 0 else 1.0
en_sign = -1.0 if pearsonr(en.predict(X_fam), u_gpls)[0] < 0 else 1.0
gcn_sign = -1.0 if pearsonr(u_gcn, u_gpls)[0] < 0 else 1.0

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         SHAP EXPLANATION LOGIC
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Setting up SHAP Explainer...")
background = np.zeros((1, 19)) # Mean-centered background since features are z-scored

def predict_for_node_i(node_idx):
    def predict_fn(X_perturbed_batch):
        # X_perturbed_batch has shape (M, 19)
        # 1. G-PLS prediction (shape M,)
        u_gpls_val = X_perturbed_batch @ gpls.x_weights_[:, 0]
        
        # 2. RF prediction (shape M,)
        u_rf_val = rf_sign * rf.predict(X_perturbed_batch)
            
        # 3. EN prediction (shape M,)
        u_en_val = en_sign * en.predict(X_perturbed_batch)
            
        # 4. GCN prediction (shape M,) - Batch parallel forward pass
        M = X_perturbed_batch.shape[0]
        X_batch_tensor = X_fam_tensor.unsqueeze(0).repeat(M, 1, 1) # (M, 455, 19)
        X_batch_tensor[:, node_idx, :] = torch.tensor(X_perturbed_batch, dtype=torch.float32)
        
        with torch.no_grad():
            u_gcn_batch = model_gcn(X_batch_tensor, adj_fc) # (M, 455, 1)
            u_gcn_val = gcn_sign * u_gcn_batch[:, node_idx, 0].numpy().flatten() # (M,)
            
        # Combine via the pre-computed gating weights for this node
        w = pred_weights[node_idx]
        pred_combined = (w[0] * u_gpls_val +
                         w[1] * u_rf_val +
                         w[2] * u_en_val +
                         w[3] * u_gcn_val)
        return pred_combined
    return predict_fn

print("Computing node-level Shapley values for all 455 brain regions...")
shap_values_matrix = np.zeros((N, 19))

# Sequential execution over 455 regions (but internal batching makes this fast!)
for i in tqdm(range(N)):
    explainer = shap.KernelExplainer(predict_for_node_i(i), background, silent=True)
    shap_vals = explainer.shap_values(X_fam[i], nsamples=100)
    shap_values_matrix[i] = shap_vals

# Save SHAP values
df_shap = pd.DataFrame(shap_values_matrix, columns=family_names, index=ns.index)
df_shap.to_csv('results/ensemble_shap_values.csv')
print("Shapley values successfully saved to results/ensemble_shap_values.csv")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOT 1: SHAP SUMMARY BAR PLOT
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Generating Plot 1: SHAP Global Summary...")
mean_abs_shap = np.mean(np.abs(shap_values_matrix), axis=0)
df_summary = pd.DataFrame({
    'Family': family_names,
    'Mean |SHAP|': mean_abs_shap
}).sort_values(by='Mean |SHAP|', ascending=False)

plt.figure(figsize=(8, 6), dpi=300)
sns.barplot(data=df_summary, y='Family', x='Mean |SHAP|', palette='viridis', hue='Family', legend=False)
plt.title("Neuropeptide Family Global Importance (Mean Absolute SHAP Value)", fontweight='bold')
plt.xlabel("Mean |SHAP| (Impact on Ensemble Behavior Prediction)", fontweight='bold')
plt.ylabel("Neuropeptide Receptor Family", fontweight='bold')
sns.despine()
plt.savefig('figs/ensemble_shap_summary.pdf', bbox_inches='tight')
plt.close()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOT 2: VIPR1 SHAP BRAINMAP
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Generating Plot 2: VIPR1 Cortical SHAP Brainmap...")
surfaces = fetch_fslr()
lh, rh = surfaces['inflated']
atlas = load_parcellation('schaefer', 400)
atlas = atlas[0] # only left hemisphere

vip_shap_ctx = shap_values_matrix[:400, vip_family_idx]

unique = np.unique(atlas)
unique = unique[1:] # discard 0
plot_data = atlas.copy()
for i in range(unique.shape[0]):
    plot_data = np.where(plot_data==unique[i], vip_shap_ctx[i], plot_data)

p = Plot(lh, views=['lateral','medial'], zoom=1.2, size=(1200, 800), brightness=0.6)
p.add_layer(plot_data, cmap=divergent_green_orange(), tick_labels=['min', 'max'])
p.build(dpi=300, save_as='figs/VIPR1_ensemble_shap_brainmap.pdf')
print("Successfully generated and saved VIPR1 SHAP brainmap to figs/VIPR1_ensemble_shap_brainmap.pdf")
