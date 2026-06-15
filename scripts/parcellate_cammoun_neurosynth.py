import os
from pathlib import Path
import pandas as pd
import numpy as np
from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import check_niimg
import warnings

warnings.filterwarnings('ignore')

def main():
    lut_path = './data/parcellations/Cammoun2012_7N_Freesurfer_Subcortex_LUT.csv'
    atlas_path = './data/parcellations/Cammoun2012_250_7N_Freesurfer_Subcortex_space-MNI152_den_1mm.nii.gz'
    out_csv = './data/neurosynth/derivatives/Cammoun2012_7N_Freesurfer_Subcortex_neurosynth.csv'
    
    print("Loading Cammoun LUT...")
    lut = pd.read_csv(lut_path)
    # Filter for scale250
    lut_scale250 = lut[lut['scale'] == 'scale250']
    regions = lut_scale250['label'].tolist()
    print(f"Loaded {len(regions)} regions for scale250.")
    
    # Get all neurosynth term directories
    ns_dir = Path('./data/neurosynth/derivatives')
    term_dirs = sorted([d for d in ns_dir.iterdir() if d.is_dir() and (d / 'z_corr-FDR_method-indep.nii.gz').exists()])
    print(f"Found {len(term_dirs)} neurosynth term maps.")
    
    # Parcellate
    data = pd.DataFrame(index=regions)
    masker = NiftiLabelsMasker(atlas_path, resampling_target='data')
    
    for term_dir in term_dirs:
        term_name = term_dir.name
        print(f"Parcellating term: {term_name}...", end='\r')
        img_path = term_dir / 'z_corr-FDR_method-indep.nii.gz'
        
        # Fit transform
        val = masker.fit_transform(check_niimg(str(img_path), atleast_4d=True)).squeeze()
        data[term_name] = val
        
    print("\nSaving parcellated neurosynth data to CSV...")
    data.to_csv(out_csv, sep=',')
    print(f"Successfully saved to {out_csv}")

if __name__ == '__main__':
    main()
