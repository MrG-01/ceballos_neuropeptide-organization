# %%
import os
import numpy as np
import pandas as pd
import scipy.stats as sstats
import seaborn as sns
import matplotlib.pyplot as plt
from neuromaps.stats import compare_images
from netneurotools.stats import get_dominance_stats
from plot_utils import divergent_green_orange
from utils import index_structure

savefig = True

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         LOAD DATA
###############################################################################
# load gene expression data
genes = pd.read_csv('results/abagen_rnaseq_interpolation_normalized.csv', index_col=0)
receptor_list = pd.read_csv('data/receptor_filtered.csv', index_col=0).index.to_list()
receptor_genes = genes[genes.columns.intersection(receptor_list)]
receptor_genes = index_structure(receptor_genes, structure='CTX')

# load receptor names from data/annotations
nt_densities = pd.read_csv('data/annotations/nt_receptor_densities_Schaefer400_TianS4_HTH.csv', index_col=0)
nt_densities = nt_densities.iloc[54:]
# strip column names, keep only first part of the name
nt_densities.columns = [name.split('_')[0] for name in nt_densities.columns]
nt_densities.rename(columns={'GABAa-bz': 'GABAa'}, inplace=True) # type: ignore

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                         DOMINANCE ANALYSIS
###############################################################################
# check if already computed
da_fn = 'results/rna_da_nt_peptides_total_dominance.npy'
if os.path.exists(da_fn):
    dom_total = np.load(da_fn)
else:
    dom_list = []
    for name in receptor_genes.columns:
        # standardize data
        X = sstats.zscore(nt_densities.values, ddof=1)
        y = sstats.zscore(receptor_genes[name].values, ddof=1)
        
        # dominance analysis
        model_metrics, model_r_sq = get_dominance_stats(X, y, n_jobs=-1)
        dom_list.append((model_metrics, model_r_sq))
    dom_total = [_[0]["total_dominance"] for _ in dom_list]
    dom_total = np.array(dom_total)
    del dom_list
    np.save(da_fn, dom_total)

# turn into relative dominance
dom_rel = (dom_total / dom_total.sum(axis=0)) * 100
# %%
# define names for plotting
peptide_names = receptor_genes.columns.values
nt_names = nt_densities.columns.values

# create domaince dataframe 
df = pd.DataFrame(dom_total.T, columns=peptide_names, index=nt_names)

# order peptide receptors by sum of dominance, i.e. R squared
receptors_by_dominance = df.sum(axis=0).sort_values(ascending=True).index
pep_idx = [df.columns.get_loc(_) for _ in receptors_by_dominance]
df = df[receptors_by_dominance]

# order nt receptors by ionotropic and metabotropic
# load nt classes
nt_classes = pd.read_csv('data/annotations/nt_receptor_classes.csv', index_col=0)
mi = nt_classes['Metab/Iono'].loc[nt_names]

# split df into two dfs and concatenate
idf = df.loc[mi[mi == 'ionotropic'].index] #type:ignore
mdf = df.loc[mi[mi == 'metabotropic'].index] #type:ignore
df = pd.concat((idf, mdf), axis=0)
nt_idx = [np.where(nt_names == _)[0][0] for _ in df.index]

# create df with relative dominance and use df order
plot_df = pd.DataFrame(dom_rel[pep_idx], index=receptors_by_dominance, 
                       columns=nt_names)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                   COMPARE METABOTROPIC AND IONOTROPIC
###############################################################################
# load receptor classes
receptor_classes = pd.read_csv('data/annotations/nt_receptor_classes.csv')

# discard transporters
receptor_classes = receptor_classes[~receptor_classes['Metab/Iono'].str.contains('trans')]
receptor_classes.set_index('protein', inplace=True)

# create df with M/I categories
dom_total_df = pd.DataFrame(dom_total.T, index=nt_densities.columns, columns=receptor_genes.columns)

# correct for uneven size of categories
m_count, i_count = receptor_classes['Metab/Iono'].value_counts().to_numpy()
categories = receptor_classes['Metab/Iono'].copy().to_frame()
categories['count'] = categories['Metab/Iono'].map({'metabotropic': m_count, 'ionotropic': i_count})
dom_total_df = dom_total_df.div(categories['count'], axis=0)

# turn into percentage contribution
dom_rel_df = dom_total_df.div(dom_total_df.sum(axis=0), axis=1) * 100

# sum contribution by category
dom_rel_df['Metab/Iono'] = dom_rel_df.index.map(categories['Metab/Iono'])
dom_category_rel_df = dom_rel_df.groupby('Metab/Iono').sum()

# test whether difference is significant
metab = dom_category_rel_df.loc['metabotropic']
iono = dom_category_rel_df.loc['ionotropic']
t, p = sstats.ttest_ind(metab, iono)

palette = divergent_green_orange(n_colors=9, return_palette=True)
bipolar = [palette[1], palette[-2]]

plt.figure(figsize=(3, 5), dpi=200)
sns.boxplot(dom_category_rel_df.T, palette=bipolar)
plt.ylabel('Average colocalization [%]')
plt.xlabel('Receptor type')
plt.tight_layout()
sns.despine(trim=True)
plt.title(f't={t:.2f} | P < 0.001')

if savefig:
    plt.savefig('figs/rna_ionotropic_metabotropic_receptors.pdf')