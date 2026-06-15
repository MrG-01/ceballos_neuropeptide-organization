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
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_centroids

# Ensure output directories exist
os.makedirs('figs', exist_ok=True)
os.makedirs('results', exist_ok=True)

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

# Load FC and SC and pad for Hypothalamus
D_hth = distance[454, :]
fc_raw = np.load('data/template_parc-Schaefer400_TianS4_desc-FC.npy')
fc_padded = np.zeros((N, N))
fc_padded[:454, :454] = np.abs(fc_raw)
closest_subcortex_idx = np.argsort(D_hth)[:4]
closest_subcortex_idx = closest_subcortex_idx[closest_subcortex_idx < 454][:3]
fc_padded[454, :454] = np.abs(fc_raw[closest_subcortex_idx, :]).mean(axis=0)
fc_padded[:454, 454] = fc_padded[454, :454]
fc_padded[454, 454] = 0.0

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         FIT BASE GCN
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

print("Fitting baseline GPLS and GCN to obtain full target behavior...")
gpls = GraphRegularizedPLS(lam=1.0).fit(X_fam, Y, distance)
t_gpls, u_gpls = gpls.transform(X_fam, Y)
t_gpls, u_gpls = t_gpls[:, 0], u_gpls[:, 0]

X_fam_tensor = torch.tensor(X_fam, dtype=torch.float32)
y_target_tensor = torch.tensor(u_gpls.reshape(-1, 1), dtype=torch.float32)

model_gcn = BrainGCN(in_dim=family_genes.shape[1], hidden_dim=64, out_dim=1, dropout=0.5)
opt = torch.optim.Adam(model_gcn.parameters(), lr=0.01, weight_decay=1e-4)

# Normalization helper that can backpropagate gradients to A
def normalize_adj_torch(A):
    A_tilde = A + torch.eye(A.shape[0])
    deg = A_tilde.sum(dim=1)
    deg_inv_sqrt = 1.0 / torch.sqrt(deg)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt) | torch.isnan(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt

# Initial train of GCN parameters using fixed adj
adj_fc_fixed = torch.tensor(normalize_adj_torch(torch.tensor(fc_padded, dtype=torch.float32)), dtype=torch.float32)
model_gcn.train()
for epoch in range(200):
    opt.zero_grad()
    loss = F.mse_loss(model_gcn(X_fam_tensor, adj_fc_fixed), y_target_tensor)
    loss.backward()
    opt.step()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         EDGE ATTRIBUTION ANALYSIS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Performing Edge Attribution via autograd on Adjacency Matrix...")
model_gcn.eval()

# Set adjacency matrix as requires_grad
A = torch.tensor(fc_padded, dtype=torch.float32, requires_grad=True)
adj_norm = normalize_adj_torch(A)

# Forward pass
preds = model_gcn(X_fam_tensor, adj_norm)
loss = F.mse_loss(preds, y_target_tensor)

# Backward pass to get dLoss/dA
loss.backward()

# Extract gradient matrix
grad_matrix = A.grad.numpy() # shape: (455, 455)

# A negative gradient means that increasing the connection strength reduces prediction error
# So we negate it to show "influence" (importance/attribution)
edge_attributions = -grad_matrix

# Save full attribution matrix
np.save('results/gcn_edge_attributions.npy', edge_attributions)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         GROUP BY YEO NETWORKS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Grouping attributions by Yeo networks...")
# Map network names
lut_matched = lut.iloc[:N]
networks = lut_matched['network'].values # shape (455,)

# Group into 8 networks
unique_networks = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'Subcortex']
network_mapping = {net: i for i, net in enumerate(unique_networks)}

net_attribution_matrix = np.zeros((8, 8))

for i in range(8):
    for j in range(8):
        mask_i = (networks == unique_networks[i])
        mask_j = (networks == unique_networks[j])
        # Average attribution of edges connecting network i and network j
        if mask_i.any() and mask_j.any():
            net_attribution_matrix[i, j] = edge_attributions[mask_i][:, mask_j].mean()

df_net_attr = pd.DataFrame(net_attribution_matrix, index=unique_networks, columns=unique_networks)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOT HEATMAP
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Plotting network edge attribution heatmap...")
plt.figure(figsize=(8, 7), dpi=300)
# Use a divergent colormap since positive means error-reducing, negative means error-increasing
sns.heatmap(df_net_attr, cmap='RdBu_r', center=0, annot=True, fmt=".2e",
            linewidths=0.5, linecolor='white', cbar_kws={'label': 'Mean Edge Attribution (Error Reduction)'})
plt.title("GCN Edge Attribution Heatmap Grouped by Yeo Networks", fontweight='bold', pad=15)
plt.ylabel("Network A", fontweight='bold')
plt.xlabel("Network B", fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('figs/gcn_edge_attribution_heatmap.pdf', bbox_inches='tight')
plt.close()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         EXTRACT TOP 50 INDIVIDUAL EDGES
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Extracting top 20 individual transmission lines...")
edges = []
for i in range(N):
    for j in range(i+1, N): # Upper triangle to avoid double counting
        edges.append({
            'node1_idx': i,
            'node2_idx': j,
            'node1_name': lut_matched.index[i],
            'node2_name': lut_matched.index[j],
            'node1_network': networks[i],
            'node2_network': networks[j],
            'attribution': edge_attributions[i, j]
        })

df_edges = pd.DataFrame(edges).sort_values(by='attribution', ascending=False)
df_edges.to_csv('results/gcn_top_attributions.csv', index=False)

print("\nTop 15 Connectome Transmission Lines (highest positive influence on prediction):")
print(df_edges.head(15)[['node1_name', 'node2_name', 'node1_network', 'node2_network', 'attribution']].to_string(index=False))
