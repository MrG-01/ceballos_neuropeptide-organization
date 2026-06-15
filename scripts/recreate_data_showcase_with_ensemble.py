import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import zscore, spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet
from plot_utils import divergent_green_orange
from utils import get_centroids

# Ensure figs directory exists
os.makedirs('figs', exist_ok=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                              LOAD DATA
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
family_genes = family_genes.T.groupby(level=0).mean().T

family_names = family_genes.columns.tolist()
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
#                         GENERATE FAMILY PREDICTIONS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Generating ensemble predictions for neuropeptide families...")
family_predictions = {}
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
    
    # Combine via gating weights (using t_gpls_rec instead of u_gpls_test to avoid leakage!)
    u_dynamic_rec = (pred_weights[:, 0] * u_gpls_rec +
                     pred_weights[:, 1] * u_rf_rec +
                     pred_weights[:, 2] * u_en_rec +
                     pred_weights[:, 3] * u_gcn_rec)
    
    # Min-max normalization of predicted scores to [0, 1] across brain regions
    u_min, u_max = u_dynamic_rec.min(), u_dynamic_rec.max()
    if u_max > u_min:
        u_norm = (u_dynamic_rec - u_min) / (u_max - u_min)
    else:
        u_norm = np.zeros_like(u_dynamic_rec)
        
    family_predictions[family_name] = u_norm

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                     ASSIGN ATLAS REGIONS TO NETWORKS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
atlas_regions = pd.read_csv('./data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv', index_col=0)
name_map = {
    'HIP': 'hippocampus',
    'THA': 'thalamus',
    'mAMY': 'amygdala',
    'lAMY': 'amygdala',
    'PUT': 'putamen',
    'aGP': 'globus-pallidus',
    'pGP': 'globus-pallidus',
    'CAU': 'caudate',
    'NAc': 'nucleus-accumbens',
    'HTH': 'hypothalamus'
}
atlas_regions['name'] = atlas_regions['name'].str.split('-').str[0].replace(name_map)
atlas_regions = atlas_regions.sort_values(by=['structure', 'network', 'name']).reset_index(drop=True)
atlas_regions = atlas_regions.sort_values('id').reset_index(drop=True)
atlas_regions['network_alt'] = atlas_regions['hemisphere'] + '_' + atlas_regions['name'].str.split('_').str[0].replace(name_map)

networks = atlas_regions['network_alt'].unique()

# Average predicted scores per network/structure
network_preds = {}
for family_name, pred_vector in family_predictions.items():
    net_values = {}
    for network in networks:
        net_values[network] = pred_vector[atlas_regions['network_alt'] == network].mean()
    network_preds[family_name] = pd.Series(net_values)

df_family_preds = pd.DataFrame(network_preds).T

# Drop right hemisphere networks
df_family_preds = df_family_preds.loc[:, ~df_family_preds.columns.str.contains('R')]

# Define target columns in exact visual layout order
network_order = [
    'L_Vis', 'L_SomMot', 'L_DorsAttn', 'L_SalVentAttn', 'L_Cont', 'L_Default', 'L_Limbic',
    'B_hypothalamus', 'L_amygdala', 'L_caudate', 'L_globus-pallidus', 'L_hippocampus', 
    'L_nucleus-accumbens', 'L_putamen', 'L_thalamus'
]
df_family_preds = df_family_preds.loc[:, network_order]

# Apply row-wise min-max normalization to the 15 network averages
for family_name in df_family_preds.index:
    row_vals = df_family_preds.loc[family_name]
    v_min, v_max = row_vals.min(), row_vals.max()
    if v_max > v_min:
        df_family_preds.loc[family_name] = (row_vals - v_min) / (v_max - v_min)
        # Scale region-level predictions using the same factors for boxplot/trace consistency
        family_predictions[family_name] = (family_predictions[family_name] - v_min) / (v_max - v_min)
    else:
        df_family_preds.loc[family_name] = 0.0
        family_predictions[family_name] = 0.0

# Map columns to clean names
clean_columns = [
    'Visual', 'Somatomotor', 'Dorsal attention', 'Ventral attention', 'Fronto-parietal', 'Default', 'Limbic',
    'Hypothalamus', 'Amygdala', 'Caudate', 'Globus pallidus', 'Hippocampus', 'Nucleus accumbens', 'Putamen', 'Thalamus'
]
df_family_preds.columns = clean_columns

# Just use the 19 families directly
df_gene_preds = df_family_preds.copy()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                     CLUSTERING (HIERARCHICAL REORDERING)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
row_linkage = linkage(df_gene_preds.values, method='average', metric='euclidean')
ordered_indices = leaves_list(row_linkage)
df_gene_preds_sorted = df_gene_preds.iloc[ordered_indices]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                     COLOR AND LEGEND CONFIGURATION
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
overview_filtered['family_clean'] = overview_filtered['family']
families_unique = sorted(list(set(overview_filtered['family_clean'])))

# Use tab20 colors for the 13 neuropeptide families
family_colors = sns.color_palette('tab20', n_colors=len(families_unique))
family_color_map = {fam: color for fam, color in zip(families_unique, family_colors)}

# Color families directly
gene_to_color = family_color_map

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                     PLOTTING PANEL a & b COMBINED
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Creating combined Matplotlib figure...")
fig = plt.figure(figsize=(15, 14), dpi=300)

# Main GridSpec: plots in top row, legend in bottom row
gs_main = gridspec.GridSpec(2, 1, height_ratios=[10, 2.5], hspace=0.18)

# Plots GridSpec (Heatmap, Dendrogram, Boxplot, Trace)
gs_plots = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[0], 
                                            width_ratios=[6, 0.8, 3.5, 1.2], wspace=0.06)

ax_heatmap = fig.add_subplot(gs_plots[0])
ax_dendro = fig.add_subplot(gs_plots[1])
ax_boxplot = fig.add_subplot(gs_plots[2])
ax_trace = fig.add_subplot(gs_plots[3])

# --- 1. HEATMAP ---
max_val = df_gene_preds_sorted.values.max()
sns.heatmap(df_gene_preds_sorted, ax=ax_heatmap, cmap=divergent_green_orange(), 
            cbar=False, vmin=0, vmax=max_val, linewidths=0.01, linecolor='white')

# Configure labels and ticks
ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=90, ha='center', fontsize=9)
ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=0, fontsize=9)

# Color y-tick labels by family color
for tick in ax_heatmap.get_yticklabels():
    gene = tick.get_text()
    tick.set_color(gene_to_color[gene])
    tick.set_weight('bold')

ax_heatmap.set_title("Neuropeptide receptor gene families (Ensemble Predictions)", 
                     fontsize=12, fontweight='bold', pad=35, loc='left')

# Add bracket headers for Cortex and Subcortex columns
# Cortex is first 7 columns, Subcortex is last 8 columns
ax_heatmap.axvline(7, color='black', linewidth=1.0, linestyle='--')
# Annotations for column brackets using axes fraction transform to avoid overlap
ax_heatmap.text(0.23, 1.05, "Cortex", transform=ax_heatmap.transAxes, ha='center', fontsize=11, fontweight='bold')
ax_heatmap.text(0.73, 1.05, "Subcortex", transform=ax_heatmap.transAxes, ha='center', fontsize=11, fontweight='bold')

# Draw horizontal lines for cortex/subcortex groupings above columns using axes fraction
ax_heatmap.annotate('', xy=(0.01, 1.02), xycoords='axes fraction', xytext=(0.46, 1.02), 
                    arrowprops=dict(arrowstyle="-", color='black', linewidth=1.5))
ax_heatmap.annotate('', xy=(0.50, 1.02), xycoords='axes fraction', xytext=(0.99, 1.02), 
                    arrowprops=dict(arrowstyle="-", color='black', linewidth=1.5))

# --- 2. DENDROGRAM ---
# Draw horizontal dendrogram aligned with heatmap rows
dendro = dendrogram(row_linkage, orientation='right', ax=ax_dendro, 
                    no_labels=True, link_color_func=lambda x: 'gray')
ax_dendro.axis('off')

# Ensure limits align with heatmap y-axis (which has 38 rows)
ax_dendro.set_ylim(0, len(ordered_indices) * 10)

# --- 3. BOXPLOT ---
# Prepare boxplot dataframe with structure labels (cortex vs subcortex vs hypothalamus)
# Each region's predicted score is parsed
lut_structures = lut_matched['structure'].values.copy()
# set hypothalamus index as 'hypothalamus'
lut_structures[454] = 'hypothalamus'

# Melt gene predictions to long format
regions_df = pd.DataFrame(family_predictions).T
regions_df.columns = [f"Region_{i}" for i in range(N)]
regions_df = regions_df.T
regions_df['structure'] = lut_structures

plot_rows = []
for fam in df_gene_preds_sorted.index:
    # Extract prediction values for all 455 regions
    fam_vals = regions_df[fam].values
    for val, struct in zip(fam_vals, lut_structures):
        plot_rows.append({'gene': fam, 'structure': struct, 'expression': val})
        
df_boxplot = pd.DataFrame(plot_rows)

# Colors
orange, yellow, green = [color for i, color in enumerate(sns.color_palette('Spectral')) if i in [1, 2, 4]]

# Separate out hypothalamus
hth_df = df_boxplot[df_boxplot['structure'] == 'hypothalamus']
df_boxplot_main = df_boxplot[df_boxplot['structure'] != 'hypothalamus']

# Render boxplots
sns.boxplot(data=df_boxplot_main, x='expression', y='gene', hue='structure', ax=ax_boxplot, 
            order=df_gene_preds_sorted.index, palette=[green, orange], hue_order=['cortex', 'subcortex'],
            showfliers=False, dodge=True, width=0.5, linewidth=0.6)

# Render pointplot for hypothalamus
sns.pointplot(data=hth_df, x='expression', y='gene', ax=ax_boxplot, color=yellow, 
              join=False, markers='o', errorbar=None, scale=0.5, order=df_gene_preds_sorted.index)

ax_boxplot.set_xlim(0, 1.05)
ax_boxplot.set_xlabel("Predicted score", fontsize=10, fontweight='bold')
ax_boxplot.set_ylabel("")
ax_boxplot.set_yticklabels([])
ax_boxplot.legend().remove()
sns.despine(ax=ax_boxplot, left=True, trim=True)
ax_boxplot.set_title("Difference across structures", fontsize=12, fontweight='bold', pad=35)

# --- 4. MEDIAN TRACE ---
# Calculate median predicted score for each structure
medians = df_boxplot.groupby(['gene', 'structure'])['expression'].median().reset_index()
medians_pivot = medians.pivot(index='gene', columns='structure', values='expression')
medians_pivot = medians_pivot.loc[df_gene_preds_sorted.index]

y_pos = np.arange(len(df_gene_preds_sorted.index))

# Plot median trace lines vertically
ax_trace.plot(medians_pivot['cortex'].values, y_pos, color=green, linewidth=1.5, label='Cortex')
ax_trace.plot(medians_pivot['hypothalamus'].values, y_pos, color=yellow, linewidth=1.5, label='Hypothalamus')
ax_trace.plot(medians_pivot['subcortex'].values, y_pos, color=orange, linewidth=1.5, label='Subcortex')

ax_trace.set_yticks(y_pos)
ax_trace.set_yticklabels(df_gene_preds_sorted.index, fontsize=9)
ax_trace.yaxis.tick_right()

# Color right-aligned gene labels
for tick in ax_trace.get_yticklabels():
    gene = tick.get_text()
    tick.set_color(gene_to_color[gene])
    tick.set_weight('bold')

ax_trace.set_xlim(-0.05, 1.05)
ax_trace.set_xlabel("Median score", fontsize=10, fontweight='bold')
ax_trace.set_title("Median trace", fontsize=12, fontweight='bold', pad=35)
ax_trace.set_ylim(-0.5, len(df_gene_preds_sorted.index) - 0.5)
ax_trace.invert_yaxis()
sns.despine(ax=ax_trace, right=False, left=True, trim=True)

# Add structure legend above the trace or boxplot
ax_boxplot.legend(
    handles=[
        plt.Line2D([0], [0], color=green, lw=4, label='Cortex'),
        plt.Line2D([0], [0], color=yellow, marker='o', ls='', label='Hypothalamus'),
        plt.Line2D([0], [0], color=orange, lw=4, label='Subcortex')
    ], 
    bbox_to_anchor=(0.5, 1.04), loc='upper center', ncol=3, frameon=False, fontsize=9
)

# --- 5. LEGEND (BOTTOM PANEL) ---
ax_legend = fig.add_subplot(gs_main[1])
ax_legend.axis('off')

# Format neuropeptide families dynamically into columns
family_genes_dict = {fam: [] for fam in families_unique}
for gene in modeled_genes:
    fam = gene_to_family[gene]
    desc = overview_filtered[overview_filtered['gene'] == gene]['description'].values[0]
    family_genes_dict[fam].append((gene, desc))

col_x = [0.01, 0.26, 0.51, 0.76]
y_start = 0.95
dy = 0.05

for idx, fam in enumerate(families_unique):
    col = idx % 4
    # Calculate y offsets within column
    y = y_start - (idx // 4) * 0.23
    
    # Render family header
    ax_legend.text(col_x[col], y, fam, fontsize=10, color=family_color_map[fam], 
                   fontweight='bold', transform=ax_legend.transAxes)
    
    # Render members
    for g_idx, (gene, desc) in enumerate(family_genes_dict[fam][:3]): # cap to top 3 members to fit
        desc_short = desc[:30] + "..." if len(desc) > 30 else desc
        member_text = f" • {gene} | {desc_short}"
        ax_legend.text(col_x[col] + 0.01, y - (g_idx + 1) * dy, member_text, 
                       fontsize=7.5, color='dimgray', transform=ax_legend.transAxes)

# Save figure
print("Saving figure to figs/recreated_data_showcase_ensemble.pdf...")
fig.savefig('figs/recreated_data_showcase_ensemble.pdf', bbox_inches='tight', dpi=300)
fig.savefig('figs/recreated_data_showcase_ensemble.png', bbox_inches='tight', dpi=300)
plt.close(fig)

print("Figure successfully generated!")
