# %%
import os
import nibabel as nib
import numpy as np
import pandas as pd
from neuromaps.nulls import burt2020
from neuromaps.datasets import fetch_atlas
from scipy.ndimage import zoom
from joblib import Parallel, delayed
from utils import index_structure

# %%
def downsample_nifti(input_file, output_file, target_voxel_size):
    # Load the original NIfTI file
    img = nib.load(input_file)
    data = img.get_fdata()
    affine = img.affine

    # Get the original voxel size from the affine matrix
    original_voxel_size = np.abs(np.diag(affine)[:3])

    # Calculate the zoom factors
    zoom_factors = original_voxel_size / target_voxel_size

    # Downsample the image data using nearest-neighbor interpolation (order=0)
    downsampled_data = zoom(data, zoom_factors, order=0)  # order=0 for nearest-neighbor interpolation

    # Ensure the downsampled data is integer type
    downsampled_data = downsampled_data.astype(np.int32)

    # Create a new affine matrix for the downsampled image
    new_affine = affine.copy()
    new_affine[:3, :3] = np.diag(target_voxel_size)

    # Create and save the new NIfTI image
    new_img = nib.Nifti1Image(downsampled_data, new_affine)
    nib.save(new_img, output_file)
    print(f"Downsampled NIfTI file saved to: {output_file}")

# Define input and output file paths
input_file = './data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz'
output_file = './data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-3mm.nii.gz'
target_voxel_size = [3.0, 3.0, 3.0]  # Target voxel size in mm

# Downsample the NIfTI file
if not os.path.exists(output_file):
    downsample_nifti(input_file, output_file, target_voxel_size)

# %% NEUROSYNTH MAPS (uncomment if you need to regenerate)
# parcellation = 'data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-3mm.nii.gz'
# neurosynth_maps = pd.read_csv('data/neurosynth_Schaefer400_TianS4.csv', index_col=0).values
# nulls = Parallel(n_jobs=12, verbose=1)(delayed(burt2020)(nmap, atlas='MNI152', density='3mm', parcellation=parcellation,
#                                                n_perm=10000, seed=0, n_jobs=1) for nmap in neurosynth_maps.T)
# nulls = np.array(nulls)
# np.save('data/neurosynth_nulls_Schaefer400_TianS4.npy', nulls)

# %% 
input_file_no_hth = './data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_noHTH_space-MNI152_den-1mm.nii.gz'
output_file_no_hth = './data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_noHTH_space-MNI152_den-3mm.nii.gz'

# Since we don't have the noHTH 1mm atlas file, let's create it if needed or check if it exists
# Usually, we can generate the noHTH version by loading the 1mm file, setting the HTH label to 0
# Let's write a robust version of that
if not os.path.exists(output_file_no_hth):
    # Load the original 1mm parcellation
    img = nib.load(input_file)
    data = img.get_fdata()
    # Hypothalamus is the last region (usually label 455). Let's set label 455 to 0 (discard it)
    data[data == 455] = 0
    # Save as noHTH 1mm file
    no_hth_1mm_img = nib.Nifti1Image(data, img.affine)
    nib.save(no_hth_1mm_img, input_file_no_hth)
    # Downsample to 3mm
    downsample_nifti(input_file_no_hth, output_file_no_hth, target_voxel_size)

# %% PEPTIDE RECEPTOR MAPS (without HTH)
print("Prefetching MNI152 3mm atlas sequentially to avoid race condition...")
fetch_atlas('MNI152', '3mm')
print("Generating receptor spatial nulls (without HTH)...")
parcellation_no_hth = 'data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_noHTH_space-MNI152_den-3mm.nii.gz'
receptor_genes_df = pd.read_csv('data/receptor_gene_expression_Schaefer2018_400_7N_Tian_Subcortex_S4.csv', index_col=0)

# only CTX-SBCTX (without HTH)
receptor_genes = index_structure(receptor_genes_df, structure='CTX-SBCTX').values

# Generate nulls in parallel using 12 jobs
nulls = Parallel(n_jobs=12, verbose=1)(delayed(burt2020)(rgene, atlas='MNI152', density='3mm', parcellation=parcellation_no_hth,
                                               n_perm=10000, seed=0, n_jobs=1) for rgene in receptor_genes.T)

nulls = np.array(nulls)
np.save('data/receptor_spatial_nulls_Schaefer400_TianS4.npy', nulls)
print("Saved data/receptor_spatial_nulls_Schaefer400_TianS4.npy")

# %% INCLUDING HTH IN PEPTIDE RECEPTOR MAPS
print("Generating receptor spatial nulls (including HTH)...")
parcellation_hth = 'data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-3mm.nii.gz'
receptor_genes_hth = receptor_genes_df.values

# Generate nulls in parallel using 12 jobs
nulls_hth = Parallel(n_jobs=12, verbose=1)(delayed(burt2020)(rgene, atlas='MNI152', density='3mm', parcellation=parcellation_hth,
                                               n_perm=10000, seed=0, n_jobs=1) for rgene in receptor_genes_hth.T)

nulls_hth = np.array(nulls_hth)
np.save('data/receptor_spatial_nulls_Schaefer400_TianS4_HTH.npy', nulls_hth)
print("Saved data/receptor_spatial_nulls_Schaefer400_TianS4_HTH.npy")
