# %%
import numpy as np
import pandas as pd
import seaborn as sns
from abagen.samples_ import update_mni_coords, mirror_samples
from abagen.io import read_ontology
from abagen.matching import AtlasTree
from abagen.images import check_atlas
from nibabel.loadsave import load as nii_load
from abagen.matching import _check_label
from abagen.allen import _get_weights
from scipy.spatial import distance_matrix
from abagen import io
from plot_utils import divergent_green_orange
from utils import scaled_robust_sigmoid

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                LOAD RNAseq DATA
#################################################################################

rnaseq_tpm = pd.read_csv("data/rnaseq/rnaseq_donor9861/RNAseqCounts.csv", index_col=0, header=None)
rnaseq_tpm = rnaseq_tpm.transpose()

rnaseq_annot = pd.read_csv("data/rnaseq/rnaseq_donor9861/SampleAnnot.csv")
rnaseq_well = rnaseq_annot['well_id']
rnaseq_tpm.index = pd.Index(rnaseq_well)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                LOAD MICROARRAY DATA
#################################################################################
microarray_expr = pd.read_csv("data/microarray/normalized_microarray_donor9861/MicroarrayExpression.csv", index_col=0, header=None)
microarray_expr = microarray_expr.transpose()
microarray_annot = pd.read_csv("data/microarray/normalized_microarray_donor9861/SampleAnnot.csv")
microarray_well = microarray_annot['well_id']
microarray_expr.index = pd.Index(microarray_well)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                 LOAD RNASEQ AND MICROARRAY FOR DONOR 10021
###################################################################################

# load rnaseq data for donor 10021
rnaseq_tpm_10021 = pd.read_csv("data/rnaseq/rnaseq_donor10021/RNAseqCounts.csv", index_col=0, header=None)
rnaseq_tpm_10021 = rnaseq_tpm_10021.transpose()
rnaseq_annot_10021 = pd.read_csv("data/rnaseq/rnaseq_donor10021/SampleAnnot.csv")
rnaseq_well_10021 = rnaseq_annot_10021['well_id']
rnaseq_tpm_10021.index = pd.Index(rnaseq_well_10021)

# load microarray data for donor 10021
microarray_expr_10021 = pd.read_csv("data/microarray/normalized_microarray_donor10021/MicroarrayExpression.csv", index_col=0, header=None)
microarray_expr_10021 = microarray_expr_10021.transpose()
microarray_annot_10021 = pd.read_csv("data/microarray/normalized_microarray_donor10021/SampleAnnot.csv")
microarray_well_10021 = microarray_annot_10021['well_id']
microarray_expr_10021.index = pd.Index(microarray_well_10021)


# %%
"""
concatenate the two microarray annotations
update the mni coordinates for the annotations
use the annotations to map well_id to mni coordinates

map to atlas or densely interpolate
"""

# concatenate the two microarray annotations and update to alleninf coordinates
microarray_annot_all = pd.concat([microarray_annot, microarray_annot_10021], axis=0).reset_index(drop=True)
microarray_annot_all = update_mni_coords(microarray_annot_all)

# create mapping from well_id to mni coordinates
map_df = pd.DataFrame(index=microarray_annot_all['well_id'])
map_df[['mni_x', 'mni_y', 'mni_z']] = microarray_annot_all[['mni_x', 'mni_y', 'mni_z']].values

# concatenate the two rnaseq annotations and join with map_df usig well_id
rnaseq_annot_all = pd.concat([rnaseq_annot, rnaseq_annot_10021], axis=0).reset_index(drop=True)
rnaseq_annot_all.set_index('well_id', drop=False, inplace=True)
rnaseq_annot_all = rnaseq_annot_all.join(map_df, how='left')

# %%
# load ontology

ontology = read_ontology("data/microarray/normalized_microarray_donor9861/Ontology.csv")
ontology.set_index('id', inplace=True)
structure_path = ontology['structure_id_path'].str.split('/').apply(pd.Series)
structure_path.index = pd.Index(ontology.index)

# from https://github.com/rmarkello/abagen/blob/dc4a007e4e902e51f97251390c8d1bbf7e58c6d3/abagen/samples_.py#L19
structure_map = pd.DataFrame(
    (('4008', 'cerebral cortex', 'cortex'),
     ('4249', 'hippocampal formation', 'subcortex/brainstem'),
     ('4275', 'cerebral nuclei', 'subcortex/brainstem'),
     ('4391', 'diencephalon', 'subcortex/brainstem'),
     ('4696', 'cerebellum', 'cerebellum'),
     ('9001', 'mesencephalon', 'subcortex/brainstem'),
     ('9131', 'pons', 'subcortex/brainstem'),
     ('9218', 'white matter', 'white matter'),
     ('9352', 'sulci & spaces', 'other'),
     ('9512', 'myelencephalon', 'subcortex/brainstem')),
    columns=['id', 'name', 'structure'])

# for each row in structure_path, find the corresponding row in structure_map
for i, row in structure_path.iterrows():
    for j, structure in structure_map.iterrows():
        # if the structure is in the path, add it to the ontology dataframe
        if structure[0] in row.values:
            ontology.at[i, 'structure'] = structure[2]
            break
        else:
            continue
        
# map ontology['structure'] to rnaseq_annot_all by joining with id and ontology_structure_id, respectively
rnaseq_annot_all.set_index('ontology_structure_id', drop=False, inplace=True)
rnaseq_annot_all = rnaseq_annot_all.join(ontology[['structure']], how='left')
ontology.reset_index(inplace=True, drop=False)

# %%
annots = rnaseq_annot_all[['well_id', 'ontology_structure_acronym', 'hemisphere', 'structure', 'mni_x', 'mni_y', 'mni_z', 'replicate_sample']]
annots.reset_index(inplace=True, drop=False)
# annots = annots[annots['replicate_sample'] == 'No']
annots_mirrored = mirror_samples(annotation=annots, ontology=ontology, swap='bidirectional').copy()
annots_mirrored.reset_index(inplace=True)


# %%
# load atlas info
atlas = nii_load('./data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz')
atlas_info = pd.read_csv('./data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_LUT.csv', index_col=0)

# validate atlas to get centroids and reload it to fit AtlasTree
atlas = check_atlas(atlas, atlas_info)
coords = pd.DataFrame(atlas.centroids).T.values
atlas = nii_load('./data/parcellations/Schaefer2018_400_7N_Tian_Subcortex_S4_space-MNI152_den-1mm.nii.gz')
tree = AtlasTree(atlas, coords=coords, atlas_info=atlas_info)

# add sample_id to annots_mirrored
annots_mirrored['sample_id'] = [f"sample_{i}" for i in range(len(annots_mirrored))]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                 ASIGN SAMPLES TO ATLAS REGIONS WITHIN 3MM
##################################################################################

# annots_mirrored = mirror_samples(annotation=annots, ontology=ontology, swap='bidirectional').copy()
# annots_mirrored.reset_index(inplace=True)
# annots_mirrored['sample_id'] = [f"sample_{i}" for i in range(len(annots_mirrored))]

for n, col in enumerate(['mni_x', 'mni_y', 'mni_z']):
    idx = np.sort(tree._volumetric[n])
    annots_mirrored[col] = idx[np.searchsorted(idx, annots_mirrored[col]) - 1]


cols = ['mni_x', 'mni_y', 'mni_z']
tol, labels = 0, np.zeros(len(annots_mirrored))
idx = np.ones(len(annots_mirrored), dtype=bool)
while tol <= 3 and np.sum(idx) > 0:
    subsamp = annots_mirrored.loc[idx]
    matches = tree.tree.query_ball_point(subsamp[cols].values, tol, p=1)
    labs = np.zeros(len(subsamp))
    for n, match in enumerate(matches):
        if len(match) > 0:
            labs[n] = tree._assign_sample(tree.atlas[match],
                                            subsamp.iloc[[n]])
    labels[idx] = labs
    idx = labels == 0
    tol += 1


# %%

def _fill_label(atlas, annotation, label, return_dist=True):
        """
        Assigns a sample in `annotation` to every node of `label` in atlas

        Parameters
        ----------
        atlas : AtlasTree
            AtlasTree object containing the atlas to be filled
        annotation : (S, 3) array_like
            At a minimum, an array of XYZ coordinates must be provided. If a
            full annotation dataframe is provided, then information from the
            data frame (i.e., on hemisphere + structural assignments of tissue
            samples) is used to constrain matching of samples (if
            `atlas.atlas_info` is not None).
        label : int
            Which label in `atlas.atlas` should be filled
        return_dist : bool, optional
            Whether to also return distance to mapped samples

        Returns
        -------
        samples : (L,) np.ndarray
            ID of sample mapped to all `L` nodes in `label` of atlas
        distance : (L,) np.ndarray
            Distances of matched samples to nodes in `label`. Only returned if
            `return_dist=True`
        """

        cols = ['mni_x', 'mni_y', 'mni_z']
        try:
            samples = io.read_annotation(annotation, copy=True)
        except TypeError:
            samples = pd.DataFrame(np.atleast_2d(annotation), columns=cols)

        missing_info = any(col not in samples.columns
                           for col in ('structure', 'hemisphere'))
        # assign samples to nearest node (i.e., vertex / voxel)
        dist, idx = atlas.tree.query(samples[cols].values, k=1)

        # now get distance between `label` nodes and assigned sample nodes
        idxs, = np.where(atlas.atlas == label)
        if not atlas.volumetric and atlas._graph is not None:
            raise ValueError('Somehow this thinks its surface, but it\'s not')
        else:
            dist = distance_matrix(atlas.coords[idxs], atlas.coords[idx])

        # check if matched samples and nodes are compatible
        if atlas.atlas_info is not None:
            labels = _check_label(atlas.atlas[idx], samples, atlas.atlas_info)
            dist[:, labels == 0] = np.inf
            # check if specified label is compatible w/nodes of matched samples
            if not missing_info:
                sh = ['structure', 'hemisphere']
                region_desc = atlas.atlas_info.loc[label, sh]
                # check if label is bilateral
                if region_desc['hemisphere'] == 'B':
                    # relax search and match samples have different structure only
                    match = region_desc['structure'] != samples['structure']
                    dist[:, np.asarray(match)] = np.inf
                else:
                    match = region_desc != samples[sh]
                    dist[:, np.asarray(np.any(match, axis=1))] = np.inf

        # get closest samples to each node of label
        closest = dist.argmin(axis=1)
        samples = samples.index[closest]

        if return_dist:
            return samples, dist[range(len(dist)), closest]
        return samples
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                       FILL IN MISSING LABELS
###################################################################################
atlas = check_atlas(atlas, atlas_info)
rnaseq_tpm_all = pd.concat([rnaseq_tpm, rnaseq_tpm_10021], axis=0)
rnaseq_tpm_all = rnaseq_tpm_all.loc[~rnaseq_tpm_all.index.duplicated(keep='first')]
# rnaseq_tpm_all.reset_index(inplace=True, drop=True)

data = []
for label in np.setdiff1d(atlas.labels, labels):
    
    # it's more cost efficient to do this check here
    if atlas.atlas_info is not None:
        cols = ['structure', 'hemisphere']
        region_desc = atlas.atlas_info.loc[label, cols]
        match = region_desc == annots_mirrored[cols]
        if not np.any(np.all(match, axis=1)) and region_desc['hemisphere'] != 'B':
            continue

    # get sample indices for every node of `lab` in `atlas`
    idx, dist = _fill_label(atlas, annots_mirrored, label, return_dist=True)
    if np.all(np.isinf(dist)):
        continue

    # get expression of selected tissue samples and weights
    dist = _get_weights(dist[:, None])


    well_id = pd.Index(annots_mirrored.loc[idx, 'well_id'])
    exp_interp = (rnaseq_tpm_all.loc[well_id] * dist).sum(axis=0) / dist.sum()
    roi = pd.Series([label], name='id')
    data.append(pd.DataFrame(exp_interp, columns=roi).T)

exp = pd.concat(data, axis=0)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                       PUTTING IT ALL TOGETHER
###################################################################################
# find the samples in labels and index them in rnaseq_tpm_all
found_structures = annots_mirrored.copy()
found_structures['atlas_region'] = labels.astype(int)
found_structures = found_structures[found_structures['atlas_region'] != 0]
found_structures = found_structures[['well_id', 'atlas_region']]

# for each atlas_region, average the corresponding well_id samples in rnaseq_tpm_all
matched_data = []
for roi in found_structures['atlas_region'].unique():
    # get the well_id samples for the current atlas_region
    well_ids = found_structures[found_structures['atlas_region'] == roi]['well_id'].values
    # get the corresponding samples in rnaseq_tpm_all
    samples = rnaseq_tpm_all.loc[well_ids]
    # average the samples
    avg_samples = samples.mean(axis=0).to_frame().T
    # rename to roi
    avg_samples.index = [roi]
    matched_data.append(avg_samples)

# concatenate the data
final = pd.concat(matched_data, axis=0)
final = pd.concat([final, exp], axis=0)

# reorder by index
final = final.loc[sorted(final.index)]

# Save unnormalized interpolation
final.to_csv('results/abagen_rnaseq_interpolation.csv')

# Compute and save normalized interpolation
normalized = pd.DataFrame(np.log1p(final.values), columns=final.columns, index=final.index)
normalized = normalized.apply(scaled_robust_sigmoid, axis=1, result_type='broadcast')
normalized = normalized.loc[:, normalized.apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25) != 0).values]
normalized = normalized.apply(scaled_robust_sigmoid, axis=0, result_type='broadcast')
normalized.to_csv('results/abagen_rnaseq_interpolation_normalized.csv')

