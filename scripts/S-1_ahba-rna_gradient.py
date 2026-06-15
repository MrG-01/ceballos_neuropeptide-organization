# %%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot_utils import divergent_green_orange
from utils import scaled_robust_sigmoid
from scipy.stats import pearsonr

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                            LOAD DATA
###############################################################################
genes = pd.read_csv('results/abagen_rnaseq_interpolation.csv', index_col=0).reset_index(drop=True)
genes = pd.DataFrame(np.log1p(genes.values), columns=genes.columns, index=genes.index)
genes = genes.apply(scaled_robust_sigmoid, axis=1, result_type='broadcast')
genes = genes.loc[:, genes.apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25) != 0).values]
genes = genes.apply(scaled_robust_sigmoid, axis=0, result_type='broadcast')

atlas_regions = pd.read_csv('data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv', index_col=0)
receptor_list = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0).columns
receptor_genes = genes[receptor_list] # type: ignore
# receptor_genes = receptor_genes.apply(scaled_robust_sigmoid, axis=0, result_type='broadcast')

# receptor_genes.index = pd.Index(atlas_regions['name'])

name_map = {'HIP': 'hippocampus',
            'THA': 'thalamus',
            'mAMY': 'amygdala',
            'lAMY': 'amygdala',
            'PUT': 'putamen',
            'aGP': 'globus-pallidus',
            'pGP': 'globus-pallidus',
            'CAU': 'caudate',
            'NAc': 'nucleus-accumbens',
            'HTH': 'hypothalamus'}

# change the name of regions in subcortex using name_map
atlas_regions['name'] = atlas_regions['name'].str.split('-').str[0].replace(name_map)

# group df first by structure, then network, and then name
atlas_regions = atlas_regions.sort_values(by=['structure', 'network', 'name']).reset_index(drop=True)

# create network label with hemisphere
atlas_regions = atlas_regions.sort_values('id').reset_index()
atlas_regions['network_alt'] = atlas_regions['hemisphere'] + '_' + atlas_regions['name'].str.split('_').str[0].replace(name_map)

# group by receptor data by network and average per network
networks = atlas_regions['network_alt'].unique()
network_genes = {network: receptor_genes[atlas_regions['network_alt'] == network].mean(axis=0) \
                 for network in networks}
network_genes = pd.DataFrame(network_genes).T

# drop all right hemisphere networks
network_genes = network_genes.loc[~network_genes.index.str.contains('R')]

# define order of networks
network_order = [f'L_{ctx_net}' for ctx_net in ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Cont', 'Default', 'Limbic' ]] + \
                [f'L_{sbctx_net}' for sbctx_net in ['amygdala', 'caudate', 'globus-pallidus', 'hippocampus', 'nucleus-accumbens', 'putamen', 'thalamus']] + \
                ['B_hypothalamus']

# reorder data and transpose to have genes as rows
network_genes = network_genes.loc[network_order].T

# plot clustermap and have the dendrogram on the same side of the xticks
clustermap = sns.clustermap(network_genes, cmap=divergent_green_orange(), col_cluster=False,
                            xticklabels=True, yticklabels=True, cbar_pos=None, figsize=(5, 11), 
                            linewidths=0.01, linecolor='white')
clustermap.figure.set_dpi(200)
plt.savefig('./figs/rnaseq_genes_heatmap.pdf')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                            PC1 COMPARISON OF ALL GENES
###############################################################################
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# load microarray data
microarray_expr = pd.read_csv("data/abagen_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv", index_col=0)

# compare PC1 of receptor genes from rnaseq and microarray
# first standardize the receptor_genes data
receptor_genes_scaled = StandardScaler().fit_transform(receptor_genes)
# then apply PCA
pca = PCA(n_components=10)
receptor_genes_pca = pca.fit_transform(receptor_genes_scaled)

# do same for microarray data
receptor_microarray = microarray_expr[receptor_genes.columns]
receptor_microarray_scaled = StandardScaler().fit_transform(receptor_microarray)
# then apply PCA
pca = PCA(n_components=10)
receptor_microarray_pca = pca.fit_transform(receptor_microarray_scaled)

# comare the first PC of both datasets
fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
sns.regplot(x=receptor_genes_pca[:, 0], y=receptor_microarray_pca[:, 0], ax=ax[0], scatter_kws={'alpha': 0},
            line_kws={'color': 'grey'}, ci=None)
sns.scatterplot(x=receptor_genes_pca[:, 0], y=receptor_microarray_pca[:, 0], ax=ax[0], alpha=0.2, 
                hue=atlas_regions['structure'])
ax[0].set_xlabel('RNAseq PC1')
ax[0].set_ylabel('Microarray PC1')

sns.regplot(x=receptor_genes_pca[:, 1], y=receptor_microarray_pca[:, 1], ax=ax[1], scatter_kws={'alpha': 0},
            line_kws={'color': 'grey'}, ci=None)
sns.scatterplot(x=receptor_genes_pca[:, 1], y=receptor_microarray_pca[:, 1], ax=ax[1], alpha=0.2, 
                hue=atlas_regions['structure'])
ax[1].set_xlabel('RNAseq PC2')
ax[1].set_ylabel('Microarray PC2')
sns.despine(trim=True)
plt.savefig('./figs/rnaseq_microarray_receptor_genes_pca_comparison.pdf')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                            APPLY PLS
###############################################################################
from pyls import behavioral_pls
from sklearn.preprocessing import StandardScaler

# load neurosynth behavioral and receptor gene data
ns = pd.read_csv('./data/neurosynth_Schaefer400_TianS4.csv', index_col=0)
receptor_genes = genes[receptor_list]

# standardize the data
X = StandardScaler().fit_transform(ns)
Y = StandardScaler().fit_transform(receptor_genes)

nlv = len(X.T) if len(X.T) < len(Y.T) else len(Y.T) # number of latent variables
lv = 0  # interested only in first latent variable
nperm = 10000 # number of permutations

# apply PLS
pls_result_rna = behavioral_pls(X, Y, n_boot=0, n_perm=0, rotate=True, permsamples=None,
                                permindices=False, test_split=0, seed=0)

pls_result_microarray = np.load('results/pls_result_Schaefer400_TianS4_HTH.npy', allow_pickle=True).item()

# compare the first latent variable between RNAseq and microarray
# start with the weights
rx, px = pearsonr(pls_result_rna['x_weights'][:, lv], pls_result_microarray['x_weights'][:, lv])
ry, py = pearsonr(pls_result_rna['y_weights'][:, lv], pls_result_microarray['y_weights'][:, lv])

# plot the weights of the first latent variable
fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
sns.regplot(x=pls_result_rna['x_weights'][:, lv], y=pls_result_microarray['x_weights'][:, lv], ax=ax[0], 
            scatter_kws={'alpha': 0.1}, line_kws={'color': 'grey'}, ci=None)

sns.regplot(x=pls_result_rna['y_weights'][:, lv], y=pls_result_microarray['y_weights'][:, lv], ax=ax[1],
            scatter_kws={'alpha': 0.1}, line_kws={'color': 'grey'}, ci=None)
ax[0].text(0.05, 0.95, f'r={rx:.2f}', transform=ax[0].transAxes, va='top', ha='left')
ax[0].set_xlabel('RNAseq behavioral weights')
ax[0].set_ylabel('Microarray behavioral weights')
ax[1].text(0.05, 0.95, f'r={ry:.2f}', transform=ax[1].transAxes, va='top', ha='left')
ax[1].set_xlabel('RNAseq receptor weights')
ax[1].set_ylabel('Microarray receptor weights')
sns.despine(trim=True)
plt.savefig('./figs/rnaseq_microarray_receptor_genes_pls_weights.pdf')

# compare scores of the first latent variable
rx, px = pearsonr(pls_result_rna['x_scores'][:, lv], pls_result_microarray['x_scores'][:, lv])
ry, py = pearsonr(pls_result_rna['y_scores'][:, lv], pls_result_microarray['y_scores'][:, lv])

fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
sns.regplot(x=pls_result_rna['x_scores'][:, lv], y=pls_result_microarray['x_scores'][:, lv], ax=ax[0],
            scatter_kws={'alpha': 0.1}, line_kws={'color': 'grey'}, ci=None)
sns.regplot(x=pls_result_rna['y_scores'][:, lv], y=pls_result_microarray['y_scores'][:, lv], ax=ax[1],
            scatter_kws={'alpha': 0.1}, line_kws={'color': 'grey'}, ci=None)
ax[0].text(0.05, 0.95, f'r={rx:.2f}', transform=ax[0].transAxes, va='top', ha='left')
ax[0].set_xlabel('RNAseq behavioral scores')
ax[0].set_ylabel('Microarray behavioral scores')
ax[1].text(0.05, 0.95, f'r={ry:.2f}', transform=ax[1].transAxes, va='top', ha='left')
ax[1].set_xlabel('RNAseq receptor scores')
ax[1].set_ylabel('Microarray receptor scores')
plt.savefig('./figs/rnaseq_microarray_receptor_genes_pls_scores.pdf')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                   COMPARE MICROARRAY AND RNASEQ WITH PLS
#################################################################################
# load data
receptor_microarray_genes = pd.read_csv("data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv", index_col=0)
nulls = np.load('data/receptor_spatial_nulls_Schaefer400_TianS4_HTH.npy')[...,:1000]
nulls = nulls.transpose(2, 1, 0)
nlv = len(X.T) if len(X.T) < len(Y.T) else len(Y.T) 
lv = 0
n_boot = 1000 
nperm = 1000 

# standardize the data
X = StandardScaler().fit_transform(receptor_microarray_genes)
Y = StandardScaler().fit_transform(receptor_genes)

pls_result_fn = 'results/rna_pls_result_Schaefer400_TianS4_HTH.npy'
if os.path.exists(pls_result_fn):
    pls_result = np.load(pls_result_fn, allow_pickle=True).item()
else:
    # behavioral PLS with gene nulls for Y
    pls_result = behavioral_pls(X, Y, n_boot=nperm, n_perm=nperm, rotate=True, permsamples=nulls,
                                permindices=False, test_split=0, seed=0, n_proc='max')
    np.save(pls_result_fn, pls_result) # type: ignore

cv = pls_result["singvals"]**2 / np.sum(pls_result["singvals"]**2)
null_singvals = pls_result['permres']['perm_singval']
cv_spins = null_singvals**2 / sum(null_singvals**2)
p = (1+sum(null_singvals[lv, :] > pls_result["singvals"][lv]))/(1+nperm)

plt.figure(figsize=(6, 5), dpi=200)
sns.boxplot(cv_spins.T * 100, color='lightgreen', zorder=1, width=0.4, linewidth=0.6,
            showfliers=False)
sns.scatterplot(x=range(nlv), y=cv*100, s=10, color='orange', linewidth=0.8, edgecolor='grey')
plt.ylabel("Covariance accounted for [%]")
plt.xlabel("Latent variables")
plt.xticks([])
plt.title(f'LV{lv+1} accounts for {cv[lv]*100:.2f}% covariance | p = {p:.4f}')
sns.despine(trim=True)
plt.savefig('figs/rna_microarray_pls_cv.pdf')

# compare scores of the first latent variable
xscore = -pls_result["x_scores"][:, lv]
yscore = -pls_result["y_scores"][:, lv]
scores = pd.DataFrame({'networks': receptor_microarray_genes.index,
                        'Microarray': xscore, 
                        'RNAseq': yscore})

atlas_info = pd.read_csv('data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv', index_col=0)
atlas_info['network'].iloc[-1] = 'Hypothalamus'
atlas_info['network'] = np.where(atlas_info['structure'] == 'cortex', 'Cortex', atlas_info['network'])
scores['network'] = atlas_info['network']

spectral = [color for i, color in enumerate(sns.color_palette('Spectral')) if i in [1,2,4]]
spectral = spectral[::-1]

fig, ax = plt.subplots(dpi=200)
sns.regplot(data=scores, x='Microarray', y='RNAseq', color='black', scatter_kws={'s': 0}, ci=None)
sns.scatterplot(data=scores, x='Microarray', y='RNAseq', hue='network', palette=spectral,
                hue_order=['Cortex', 'Hypothalamus', 'Subcortex'], s=50)
ax.set_xlabel('Microarray score')
ax.set_ylabel('RNAseq score')
sns.despine()
plt.legend(frameon=False, title='Network')
plt.savefig('figs/rna_microarray_pls_scores.pdf')

