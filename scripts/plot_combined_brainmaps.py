import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from plot_utils import divergent_green_orange
from surfplot import Plot
from brainspace.datasets import load_parcellation
from neuromaps.datasets import fetch_fslr
from utils import get_centroids
import io
from PIL import Image, ImageOps

# Ensure figs directory exists
os.makedirs('figs', exist_ok=True)

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

family_genes = receptor_genes.copy()
family_genes.columns = [gene_to_family.get(c, c) for c in family_genes.columns]
family_genes = family_genes.groupby(by=family_genes.columns, axis=1).mean()

# Family names
family_names = family_genes.columns.tolist()

# The 15 key receptors to map
receptors = [
    'ADIPOR2',
    'GRPR',
    'CALCRL',
    'CCKBR',
    'EDNRB',
    'NPY1R',
    'GALR1',
    'VIPR1',
    'RXFP1',
    'NTSR1',
    'NPR2',
    'OPRK1',
    'SSTR1',
    'OXTR',
    'GHR'
]

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
#                         LOAD SURFACE AND PARCELLATION
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
surfaces = fetch_fslr()
lh, rh = surfaces['inflated']
atlas = load_parcellation('schaefer', 400)
atlas = atlas[0] # only left hemisphere
unique = np.unique(atlas)[1:] # discard 0

# Helper function to crop whitespace from PIL Image
def crop_whitespace(img, padding=5):
    img_gray = img.convert('L')
    img_inv = ImageOps.invert(img_gray)
    bbox = img_inv.getbbox()
    if bbox:
        left = max(0, bbox[0] - padding)
        top = max(0, bbox[1] - padding)
        right = min(img.width, bbox[2] + padding)
        bottom = min(img.height, bbox[3] + padding)
        return img.crop((left, top, right, bottom))
    return img

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         GENERATE AND CROP BRAINMAPS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
images = {}
print("Generating individual brain maps in memory...")
for family_name in family_names:
    family_idx = family_names.index(family_name)
    
    # Isolate family features
    X_rec = np.zeros_like(X_fam)
    X_rec[:, family_idx] = X_fam[:, family_idx]
    X_rec_tensor = torch.tensor(X_rec, dtype=torch.float32)
    
    # Predict from base models
    u_gpls_rec = X_rec @ gpls.x_weights_[:, 0]
    
    u_rf_rec = rf.predict(X_rec)
    if pearsonr(rf.predict(X_fam), u_gpls)[0] < 0: u_rf_rec = -u_rf_rec
    
    u_en_rec = en.predict(X_rec)
    if pearsonr(en.predict(X_fam), u_gpls)[0] < 0: u_en_rec = -u_en_rec
    
    with torch.no_grad():
        u_gcn_rec = model_gcn(X_rec_tensor, adj_fc).numpy().flatten()
    if pearsonr(u_gcn, u_gpls)[0] < 0: u_gcn_rec = -u_gcn_rec
    
    # Combine via gating weights
    u_dynamic_rec = (pred_weights[:, 0] * u_gpls_rec +
                     pred_weights[:, 1] * u_rf_rec +
                     pred_weights[:, 2] * u_en_rec +
                     pred_weights[:, 3] * u_gcn_rec)
    
    # Slice first 400 cortical regions
    rec_ctx_pred = u_dynamic_rec[:400]
    
    # Populate atlas regions
    plot_data = atlas.copy()
    for i in range(unique.shape[0]):
        plot_data = np.where(plot_data==unique[i], rec_ctx_pred[i], plot_data)
        
    # Plot using surfplot
    p = Plot(lh, views=['lateral','medial'], zoom=1.2, size=(1200, 800), brightness=0.6)
    p.add_layer(plot_data, cmap=divergent_green_orange(), tick_labels=['min', 'max'])
    
    # Build to memory buffer
    fig_sp = p.build(dpi=200)
    buf = io.BytesIO()
    fig_sp.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.0)
    plt.close(fig_sp)
    buf.seek(0)
    
    # Open PIL Image and crop whitespace
    img_pil = Image.open(buf)
    img_cropped = crop_whitespace(img_pil)
    images[family_name] = img_cropped
    print(f"Captured and cropped map for {family_name}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         CREATE COMBINED GRID PLOT
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Creating combined 5x4 grid figure...")
fig, axes = plt.subplots(5, 4, figsize=(18, 18), dpi=300)

for idx, family_name in enumerate(family_names):
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]
    
    img_cropped = images[family_name]
    
    # Show cropped brain map image
    ax.imshow(img_cropped)
    ax.axis('off')
    
    # Title showing family name
    ax.set_title(family_name, fontsize=11, fontweight='bold', pad=4)

# Hide the 20th subplot (row 4, col 3) since we only have 19 families
axes[4, 3].axis('off')

# Adjust spacing and layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.suptitle("Ensemble-Predicted Behavioral Projections for 19 Receptor Families", 
             fontsize=20, fontweight='bold', y=0.98)

# Save combined figure
print("Saving combined plots to figs/...")
fig.savefig('figs/combined_ensemble_brainmaps.pdf', bbox_inches='tight', dpi=300)
fig.savefig('figs/combined_ensemble_brainmaps.png', bbox_inches='tight', dpi=300)
plt.close(fig)

print("Combined ensemble brainmaps generated successfully!")
