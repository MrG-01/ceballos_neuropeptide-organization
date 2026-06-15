# %%
import contextlib
import json
import os
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import requests
from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import check_niimg
from nibabel.loadsave import save as niisave

from nimare.dataset import Dataset
from nimare.meta.cbma.ale import ALE
from nimare.correct import FDRCorrector
from nimare.io import convert_neurosynth_to_dataset
from nimare.extract import fetch_neurosynth

from urllib.request import urlopen
import sys

# /sigh
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

atlas = './data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz'
labels = pd.read_csv('./data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv')['name'].to_list()

# %%
# this is where the raw and parcellated data will be stored
NSDIR = Path('./data/neurosynth/raw').resolve()
PARDIR = Path('./data/neurosynth/derivatives').resolve()

# these are the images from the neurosynth analyses we'll save
IMAGES = 'z_corr-FDR_method-indep'


def fetch_ns_data(directory):
    """ Fetches NeuroSynth database + features to `directory` """
    directory = Path(directory)

    # if not already downloaded, download the NS data and unpack it
    database = directory / 'neurosynth_dataset.pkl.gz'
    if not database.exists():
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            neurosynth_db = fetch_neurosynth(data_dir=directory, return_type='dataset', source='abstract', vocab='terms')

        neurosynth_dset = neurosynth_db[0]
        neurosynth_dset.save(database)
    else:
        neurosynth_dset = Dataset.load(database)
    
    return neurosynth_dset


def run_meta_analyses(dset, use_features=None, outdir=None, sample_size=30):
    """ Runs NiMARE-style meta-analysis based on `database` and `features` """
    if outdir is None:
        outdir = NSDIR
    outdir = Path(outdir)
    
    # ensure 'sample_sizes' field exists
    if 'sample_sizes' not in dset.metadata.columns:
        dset.metadata['sample_sizes'] = sample_size

    # if we only want a subset of the features take the set intersection
    if use_features is not None:
        labels = dset.get_labels()
        
        # get only last part of the label as matching term
        terms = [l.split('__')[-1] for l in labels]
        terms = set(terms) & set(use_features)

        # find matching labels
        features = set([l for l in labels if l.split('__')[-1] in terms])
    else:
        features = set(dset.get_labels())
    pad = max([len(f) for f in features])

    generated = []
    for word in sorted(features):
        msg = f'Running meta-analysis for term: {word:<{pad}}'
        print(msg, end='\r', flush=True)

        # run meta-analysis + save specified outputs (only if they don't exist)
        path = outdir / word.split('__')[-1]
        path.mkdir(exist_ok=True)
        if not (path / f'{IMAGES}.nii.gz').exists():
            # find studies with term
            ids = dset.get_studies_by_label(word)
            subset_dset = dset.slice(ids)
            
            # run meta-analysis
            ma = ALE()
            result = ma.fit(subset_dset)
            corrector = FDRCorrector(alpha=0.01, method='indep')
            corrected_results = corrector.transform(result)
            nii = corrected_results.get_map(IMAGES)
            nii.to_filename(path / f'{IMAGES}.nii.gz')

        # store MA path
        generated.append(path)

    print(' ' * len(msg) + '\b' * len(msg), end='', flush=True)

    return generated


def parcellate_meta(outputs, annots, fname, regions):
    # empty dataframe to hold our parcellated data
    data = pd.DataFrame(index=regions)
    mask = NiftiLabelsMasker(annots, resampling_target='data')

    for outdir in outputs:
        cdata = []
        mgh = outdir / 'z_corr-FDR_method-indep.nii.gz'

        cdata.append(mask.fit_transform(
            check_niimg(mgh.__str__(), atleast_4d=True)).squeeze())

        # store it in the dataframe
        data = data.assign(**{outdir.name: np.hstack(cdata)})

    # now we save the dataframe
    data.to_csv(fname, sep=',')
    return fname


if __name__ == '__main__':
    NSDIR.mkdir(parents=True, exist_ok=True)
    PARDIR.mkdir(parents=True, exist_ok=True)

    # load the 125 cognitive terms from the paper's Excel file
    excel_path = '/Users/matthewgruner/Projects/zandawala/datasets/Ceballos2026/41593_2026_2236_MOESM3_ESM (1).xlsx'
    terms_df = pd.read_excel(excel_path, sheet_name='fig5b_terms')
    term_list = terms_df['term'].astype(str).tolist()
    # convert terms to lowercase for matching Neurosynth database keys
    term_list = [t.lower() for t in term_list]

    print(f"Loaded {len(term_list)} terms from Excel file.")

    # fetch Neurosynth database
    dset = fetch_ns_data(NSDIR)
    
    # run relevant NS meta-analyses only for these terms
    generated = run_meta_analyses(dset, term_list, outdir=PARDIR)
    
    # parcellate data and save to target directory
    parcellate_meta(generated, atlas,
                    './data/neurosynth_Schaefer400_TianS4.csv',
                    regions=labels)
    print("Successfully parcellated Neurosynth maps and saved to ./data/neurosynth_Schaefer400_TianS4.csv")
