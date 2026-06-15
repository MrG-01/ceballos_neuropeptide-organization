import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from scipy.stats import zscore, spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform, cdist
from brainspace.gradient import GradientMaps
from utils import get_centroids

# Ensure figs directory exists
os.makedirs('figs', exist_ok=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         LOAD DATA
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load receptor genes (shape 455 x 38)
receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
# Load parcellation LUT
lut = pd.read_csv('data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv', index_col=0)

# Load FC and SC templates (454 x 454)
fc = np.load('data/template_parc-Schaefer400_TianS4_desc-FC.npy')
sc = np.load('data/template_parc-Schaefer400_TianS4_desc-SC.npy')

# Compute gradients on FC
gm = GradientMaps(n_components=2, approach='dm', kernel='normalized_angle')
gm.fit(fc)
g1 = gm.gradients_[:, 0]
g2 = gm.gradients_[:, 1]

# Align LUT and data (exclude HTH for gradient-based analyses)
# First 454 rows of LUT align with FC/SC matrices and the first 454 regions of receptor genes
lut_454 = lut.iloc[:454].copy()
receptor_genes_454 = receptor_genes.iloc[:454].copy()

# Compute HTH Centroid and Euclidean distances to HTH
img = nib.load('data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz')
labels = np.unique(img.get_fdata())[1:] # discard background 0
centroids = get_centroids(img, labels=labels)
hth_centroid = centroids[-1] # last one is HTH (label 455)
distances_to_hth = cdist(centroids[:-1], [hth_centroid]).flatten() # distance to all other 454 regions

# Add spatial/gradient metrics to LUT
lut_454['g1'] = g1
lut_454['g2'] = g2
lut_454['fc_strength'] = fc.sum(axis=1)
lut_454['sc_degree'] = (sc > 0).sum(axis=1)
lut_454['distance_to_hth'] = distances_to_hth
lut_454['VIPR1'] = receptor_genes_454['VIPR1'].values

# Define Network Order for Plots (Hierarchical order Vis -> Default + Subcortex)
network_order = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Subcortex', 'Limbic', 'Cont', 'Default']
# Create color palette for networks
spectral_palette = sns.color_palette("Spectral", len(network_order))
network_colors = dict(zip(network_order, spectral_palette))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOT 1: RIDGELINE DENSITY PLOT
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Generating Plot 1: Ridgeline Density Plot...")
# Group by network and create a ridgeline plot of VIPR1 expression
df_ridge = lut_454[['network', 'VIPR1']].copy()
# Z-score VIPR1 for clean visualization
df_ridge['VIPR1_z'] = zscore(df_ridge['VIPR1'])

# Set style
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
g = sns.FacetGrid(df_ridge, row="network", hue="network", aspect=9, height=0.6, 
                  palette=network_colors, row_order=network_order)

# Map KDE plots to show densities
g.map(sns.kdeplot, "VIPR1_z", bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "VIPR1_z", clip_on=False, color="white", lw=2, bw_adjust=.5)

# Add reference line
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# Define a function to label each row
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "VIPR1_z")

# Set overlap between rows
g.figure.subplots_adjust(hspace=-.25)

# Remove axes details
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)
plt.xlabel("VIPR1 Expression (Z-score)", fontweight="bold")
plt.suptitle("VIPR1 Expression Ridgeline Profiles Across Hierarchical Networks", fontsize=10, fontweight="bold", y=0.98)
plt.savefig('figs/vipr1_ridgeline.pdf', bbox_inches='tight')
plt.close()

# Reset seaborn theme for subsequent plots
sns.set_theme(style="ticks")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOT 2: GRADIENT TRAJECTORY PLOT
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Generating Plot 2: Gradient Trajectory Plot...")
plt.figure(figsize=(7, 5), dpi=300)
# Plot regions as scatter colored by network
sns.scatterplot(data=lut_454, x='g1', y='VIPR1', hue='network', palette=network_colors, 
                hue_order=network_order, s=35, alpha=0.8, edgecolor='none')

# Add smooth LOESS-like fit line (order 2 polynomial)
sns.regplot(data=lut_454, x='g1', y='VIPR1', scatter=False, color='black', 
            line_kws={'linewidth': 2.5, 'linestyle': '--'}, ci=95)

plt.xlabel("Sensory-to-Association Gradient 1 (g1)", fontweight="bold")
plt.ylabel("VIPR1 Expression", fontweight="bold")
plt.title("VIPR1 Expression Along the Functional Sensory-Association Gradient", fontsize=10, fontweight="bold")
plt.legend(frameon=True, facecolor='white', edgecolor='none', title='Network', bbox_to_anchor=(1.05, 1), loc='upper left')
sns.despine()
plt.savefig('figs/vipr1_trajectory.pdf', bbox_inches='tight')
plt.close()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOT 3: YEO NETWORK RADAR CHART
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Generating Plot 3: Yeo Network Radar Chart...")
# Compute median expression of VIPR1 per network
network_medians = lut_454.groupby('network')['VIPR1'].median().reindex(network_order).values

# Number of variables/networks
categories = network_order
N = len(categories)

# We want the plot to be circular, so we loop back to the start
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
values = list(network_medians)
values += values[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), dpi=300)

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories, color='grey', size=8)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks(color="grey", size=7)
plt.ylim(0, max(values) * 1.1)

# Plot data
ax.plot(angles, values, color='#4C72B0', linewidth=2, linestyle='solid')
ax.fill(angles, values, color='#4C72B0', alpha=0.3)

plt.title("VIPR1 Median Expression Profile Across Functional Networks", size=10, fontweight='bold', y=1.1)
plt.savefig('figs/vipr1_radar.pdf', bbox_inches='tight')
plt.close()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         PLOT 4: RECEPTOR-GRADIENT CORRELATION CLUSTERMAP
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("Generating Plot 4: Receptor-Gradient Correlation Clustermap...")

# Compute correlation of each receptor with g1, g2, fc_strength, sc_degree, and distance_to_hth
features = ['g1', 'g2', 'fc_strength', 'sc_degree', 'distance_to_hth']
corr_data = []

for rgene in receptor_genes_454.columns:
    r_vals = []
    for feat in features:
        r_corr, _ = spearmanr(receptor_genes_454[rgene], lut_454[feat])
        r_vals.append(r_corr)
    corr_data.append(r_vals)

df_corr = pd.DataFrame(corr_data, index=receptor_genes_454.columns, columns=['Gradient 1', 'Gradient 2', 'FC Strength', 'SC Degree', 'HTH Distance'])

# Plot clustermap
g = sns.clustermap(df_corr, cmap='RdBu_r', center=0, annot=True, fmt=".2f",
                   figsize=(8, 12), cbar_kws={'label': 'Spearman Correlation'},
                   linewidths=0.5, linecolor='white')

# Rotate row labels for readability
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.suptitle("Neuropeptide Receptor Correlation with Hierarchical Brain Metrics", fontsize=12, fontweight='bold', y=1.02)
plt.savefig('figs/receptors_gradient_clustermap.pdf', bbox_inches='tight')
plt.close()

print("All 4 hierarchy plots generated successfully under figs/ directory!")
