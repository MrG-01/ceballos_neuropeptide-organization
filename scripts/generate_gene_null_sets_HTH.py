import os
import nibabel as nib
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from utils import gene_null_set, get_centroids

def main():
    out_file = 'data/gene_null_sets_Schaefer400_TianS4_HTH.npy'
    if os.path.exists(out_file):
        print(f"{out_file} already exists.")
        return
        
    print("Loading gene expression data...")
    receptor_genes = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
    all_genes = pd.read_csv('data/abagen_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)
    peptide_list = pd.read_csv('data/gene_list.csv')['Gene']
    non_peptide_genes = all_genes.T[~all_genes.columns.isin(peptide_list)].T

    print("Loading centroids and computing distance matrix...")
    img = nib.load('data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz')
    labels = np.unique(img.get_fdata())[1:] # discard background 0
    centroids = get_centroids(img, labels=labels)
    distance = squareform(pdist(centroids))

    # We will generate 100 permutations for a fast but reasonable null distribution
    nperm = 100
    print(f"Generating gene null sets (n_permutations={nperm})...")
    nulls = gene_null_set(receptor_genes, non_peptide_genes, distance, n_permutations=nperm, 
                          n_jobs=-1, seed=0)

    nulls = np.array(nulls)
    np.save(out_file, nulls)
    print(f"Saved {out_file} with shape {nulls.shape}")

if __name__ == '__main__':
    main()
