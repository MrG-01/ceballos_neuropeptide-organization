# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from abagen.samples_ import update_mni_coords
from glob import glob
from utils import scaled_robust_sigmoid
from sklearn.decomposition import PCA
from plot_utils import divergent_green_yellow_orange
from mpl_toolkits.mplot3d import Axes3D

savefig = True

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                              LOAD DATA
###################################################################################

donor_ids = []
for path in glob("data/microarray/normalized_microarray_donor*"):
    path = path.split('/')[-1]
    donor_id = int(path.split('_')[-1][5:])
    donor_ids.append(donor_id)
donor_ids = sorted(donor_ids)

microarray_exprs = []
microarray_annots = []
for donor in donor_ids:
    microarray_expr = pd.read_csv(f"data/microarray/normalized_microarray_donor{donor}/MicroarrayExpression.csv", 
                             index_col=0, header=None)
    microarray_expr = microarray_expr.transpose()
    
    # normalize the expression values using robust sigmoid
    microarray_expr = microarray_expr.apply(scaled_robust_sigmoid, axis=1, result_type='broadcast')
    microarray_expr = microarray_expr.loc[:, microarray_expr.apply(
                           lambda x: np.percentile(x, 75) - np.percentile(x, 25) != 0).values]
    microarray_expr = microarray_expr.apply(scaled_robust_sigmoid, axis=0, result_type='broadcast')
    
    microarray_exprs.append(microarray_expr)
    
    microarray_annot = pd.read_csv(f"data/microarray/normalized_microarray_donor{donor}/SampleAnnot.csv")
    microarray_well = microarray_annot['well_id']
    microarray_expr.index = pd.Index(microarray_well)
    microarray_annots.append(microarray_annot)
    
# update the MNI coordinates in microarray_annots
microarray_annots = pd.concat([update_mni_coords(annot) 
                              for annot in microarray_annots], 
                              axis=0).reset_index(drop=True)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                     LOOK FOR HYPOTHALAMIC NUCLEI
####################################################################################

# look for the following structures in structure_name
structures = ['ventromedial hypothalamic nucleus',
              'perifornical nucleus',
              'posterior hypothalamic area',
              'dorsomedial hypothalamic nucleus',
              'medial mammillary nucleus',
              'lateral mammillary nucleus',
              'lateral hypothalamic area, mammillary region',
              'tuberomammillary nucleus']

# in microarray_annots, find the rows where structure_name is in structures
mask = microarray_annots['structure_name'].str.contains('|'.join(structures), case=False, na=False)

# filter the microarray_annots dataframe using the mask
microarray_annots_filtered = microarray_annots[mask].reset_index(drop=True)

# find the well_ids from microarray_annots_filtered in microarray_exprs
microarray_exprs_filtered = []
for expr in microarray_exprs:
    mask = expr.index.isin(microarray_annots_filtered['well_id'])
    filtered_expr = expr.loc[mask]
    microarray_exprs_filtered.append(filtered_expr)
    
# concatenate the filtered expressions
microarray_exprs_filtered = pd.concat(microarray_exprs_filtered, axis=0)

# use structure_name from microarray_annots_filtered to set the index of microarray_exprs_filtered
# match using well_id
names = microarray_annots_filtered.set_index('well_id').loc[microarray_exprs_filtered.index, 'structure_name']
microarray_exprs_filtered.index = pd.Index(names)

# average the expression values by structure_name
microarray_exprs_avg = microarray_exprs_filtered.groupby(microarray_exprs_filtered.index).mean()

# use Probes.csv to get the gene names
probes = []
for donor in donor_ids:
    probes_df = pd.read_csv(f"data/microarray/normalized_microarray_donor{donor}/Probes.csv", index_col=0)
    probes.append(probes_df)
    
probes = pd.concat(probes, axis=0).reset_index()
probe_id_to_gene_symbol = dict(zip(probes['probe_id'], probes['gene_symbol']))

# map the probe_id to gene_symbol in microarray_exprs_avg
microarray_exprs_avg.columns = microarray_exprs_avg.columns.map(probe_id_to_gene_symbol)

# average across gene columns
microarray_exprs_avg = microarray_exprs_avg.T.groupby(microarray_exprs_avg.columns).mean().T

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                        LOOK FOR RECEPTOR EXPRESSION 
####################################################################################
# load receptors_filtered.csv
receptors = pd.read_csv('data/receptor_filtered.csv')['gene'].tolist()
receptor_expr = microarray_exprs_avg.loc[:,microarray_exprs_avg.columns.isin(receptors)].copy()

receptor_coords = []
acronyms = []
for structure in receptor_expr.index:
    # find the rows in microarray_annots_filtered where structure_name matches
    mask = microarray_annots_filtered['structure_name'] == structure
    coords = microarray_annots_filtered.loc[mask, ['mni_x', 'mni_y', 'mni_z']].mean().values
    acronyms.append(microarray_annots_filtered.loc[mask, 'structure_acronym'].values[0])
    receptor_coords.append(coords)
# create a DataFrame with the coordinates and set the index to receptor_expr.index
receptor_coords = pd.DataFrame(receptor_coords, index=receptor_expr.index, columns=['mni_x', 'mni_y', 'mni_z'])
# add the acronyms to the receptor_coords DataFrame
receptor_coords['acronym'] = acronyms

# keep only the structures on the left hemisphere (mni_x < 0)
receptor_expr = receptor_expr[receptor_coords['mni_x'] < 0]
receptor_coords = receptor_coords[receptor_coords['mni_x'] < 0]

# PCA on the receptor expression values
pca = PCA(n_components=8)
pca.fit(receptor_expr)
# use loadings_ to get the coordinates in the PCA space
pca_coords = pca.transform(receptor_expr)[:,:3] # x, y, z
pca_coords = pd.DataFrame(pca_coords, index=receptor_expr.index, columns=['pca_x', 'pca_y', 'pca_z'])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                PLOT NUCLEI
#####################################################################################

x, y, z = pca_coords[['pca_x', 'pca_y', 'pca_z']].values.T
acronyms = receptor_coords['acronym'].values

n_plots = len(receptor_expr.T)
fig, axs = plt.subplots(7, 6, figsize=(10, 10), dpi=200,
                        subplot_kw={'projection': '3d'})

# Set the desired viewing angles
elev = 0  # elevation angle in degrees
azim = -90  # azimuthal angle in degrees
roll = 45

for row, ax in zip(receptor_expr.T.iterrows(), axs.flatten()):    
    gene = row[0]
    expr = row[1].values

    # scatterplot the expression values
    sc = ax.scatter(x, y, z, c=expr, cmap=divergent_green_yellow_orange(), s=40)
    # Set the 3D view angle
    ax.view_init(elev=elev, azim=azim)

    # # add a text label with each dot
    # for i, acronym in enumerate(acronyms):
    #     ax.text(x[i], y[i], z[i], f"{acronym}", fontsize=6, ha='center', va='center')
    
    # set color limits to 0-1
    sc.set_clim(0, 1)
    # set the title and labels
    # make sure to have the title close to the figure
    ax.set_title(gene, y=0.85)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # remove the axis lines
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # remove axis background and grid
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)

axs.flatten()[-1].remove()
axs.flatten()[-2].remove()
axs.flatten()[-3].remove()
axs.flatten()[-4].remove()

sns.despine(fig=fig, top=True, right=True, left=False, bottom=False)
plt.subplots_adjust(wspace=-0.1, hspace=-0.2)

if savefig:
    plt.savefig('figs/hth_nuclei_receptor_expression.pdf')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                PLOT HEATMAP
#####################################################################################

# order receptor expression
# first sort receptors by order from figure 2 
fig_order = np.load('results/gene_expression_cluster_order.npy')[::-1]

# order the structures by their first PCA component
pca_order = pca_coords.sort_values(by='pca_x', ascending=False).index

# order dataframe
rececptor_expr_ordered = receptor_expr.loc[pca_order].iloc[:, fig_order].copy()

# plot clustermap of the receptor expression values
fig, ax = plt.subplots(figsize=(3, 8), dpi=200)
sns.heatmap(rececptor_expr_ordered.T, cmap=divergent_green_yellow_orange(),
            linewidths=0.01, linecolor='white', square=True, cbar=None, ax=ax)
ax.set_xticks([])
ax.set_xlabel('')
ax.set_ylabel('')

if savefig:
    plt.savefig('figs/hth_nuclei_receptor_expression_heatmap.pdf')