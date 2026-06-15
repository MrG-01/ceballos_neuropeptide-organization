import os
from pathlib import Path
import warnings
import pandas as pd
from nimare.dataset import Dataset
from nimare.meta.cbma.ale import ALE
from nimare.correct import FDRCorrector

warnings.filterwarnings('ignore')

def main():
    db_path = './data/neurosynth/raw/neurosynth_dataset.pkl.gz'
    print("Loading Neurosynth dataset...")
    dset = Dataset.load(db_path)
    
    # Check labels in dataset matching 'food' and 'eating'
    labels = dset.get_labels()
    
    food_label = None
    eating_label = None
    for lbl in labels:
        term = lbl.split('__')[-1]
        if term == 'food':
            food_label = lbl
        elif term == 'eating':
            eating_label = lbl
            
    print(f"Matched labels: food -> {food_label}, eating -> {eating_label}")
    
    # Run meta-analysis for food
    if food_label:
        out_food = './data/food_term.nii.gz'
        if not os.path.exists(out_food):
            print("Running meta-analysis for 'food'...")
            ids = dset.get_studies_by_label(food_label)
            print(f"Found {len(ids)} studies matching 'food'")
            subset_dset = dset.slice(ids)
            
            # Ensure 'sample_sizes' field exists
            if 'sample_sizes' not in subset_dset.metadata.columns:
                subset_dset.metadata['sample_sizes'] = 30
                
            ma = ALE()
            result = ma.fit(subset_dset)
            corrector = FDRCorrector(alpha=0.01, method='indep')
            corrected_results = corrector.transform(result)
            nii = corrected_results.get_map('z_corr-FDR_method-indep')
            nii.to_filename(out_food)
            print(f"Saved 'food' term map to {out_food}")
        else:
            print(f"{out_food} already exists.")
            
    # Copy/Run meta-analysis for eating
    out_eating = './data/eating_term.nii.gz'
    if not os.path.exists(out_eating):
        eating_src = './data/neurosynth/derivatives/eating/z_corr-FDR_method-indep.nii.gz'
        if os.path.exists(eating_src):
            print(f"Copying existing 'eating' map from {eating_src}")
            import shutil
            shutil.copy(eating_src, out_eating)
        elif eating_label:
            print("Running meta-analysis for 'eating'...")
            ids = dset.get_studies_by_label(eating_label)
            print(f"Found {len(ids)} studies matching 'eating'")
            subset_dset = dset.slice(ids)
            if 'sample_sizes' not in subset_dset.metadata.columns:
                subset_dset.metadata['sample_sizes'] = 30
            ma = ALE()
            result = ma.fit(subset_dset)
            corrector = FDRCorrector(alpha=0.01, method='indep')
            corrected_results = corrector.transform(result)
            nii = corrected_results.get_map('z_corr-FDR_method-indep')
            nii.to_filename(out_eating)
            print(f"Saved 'eating' term map to {out_eating}")
    else:
        print(f"{out_eating} already exists.")

if __name__ == '__main__':
    main()
