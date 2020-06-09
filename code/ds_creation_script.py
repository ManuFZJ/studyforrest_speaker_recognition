#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
from pathlib import Path

import numpy as np
from numpy.testing.decorators import skipif
import mvpa2.suite as mvpa
from mvpa2.measures import rsa

import utility_functions as uf


# In[2]:


# Wider Output
np.set_printoptions(edgeitems=20)

# Project Directory
project_dir = Path.cwd().parents[0]

# Get Participant Files
data_dir = project_dir.joinpath("data", "studyforrest-data-aligned")
ao_files = uf.get_files(data_dir, '*aomovie*nii.gz')
av_files = uf.get_files(data_dir, '*avmovie*nii.gz')
all_files = ao_files + av_files

# Get Annotation Files
anno_dir = project_dir.joinpath("data", "tmp", "speech_anno")
ao_anno_files = uf.get_files(anno_dir.joinpath("aomovie"), '*.tsv')
av_anno_files = uf.get_files(anno_dir.joinpath("avmovie"), '*.tsv')

# Get ROI Files
mask_dir = project_dir.joinpath("data", "tmp", "masks_final_approach", "finished (inc. brain mask overlap)")
roi_masks_list = uf.get_files(mask_dir, '*.nii.gz')
roi_masks_list = [mask for mask in roi_masks_list if "all" not in mask]

# Reference Space + Warp Files
template_dir = project_dir.joinpath("data", "studyforrest-data-templatetransforms")
ref = template_dir.joinpath("templates", "grpbold3Tp2", "brain.nii.gz")
warp_files = uf.get_files(template_dir, '*subj2tmpl_warp*.nii.gz')

# Check if all components are good to go
path_list = [project_dir, data_dir, anno_dir, template_dir, ref, mask_dir]
file_lists = [ao_files, av_files, ao_anno_files, av_anno_files, warp_files, roi_masks_list]

check = uf.check_all_components(path_list, file_lists)
print "{}\n".format(check[0])
print "{}\n".format(check[1])


# In[3]:


# DS
files_to_inc = sys.argv[1]

if "av" in files_to_inc and "ao" in files_to_inc:
    data_files = all_files[0:2]
elif files_to_inc == "av":
    data_files = av_files[0:2]
elif files_to_inc == "ao":
    data_files = ao_files[0:2]
    
key = "preprocessed_df_{}.hdf5".format(files_to_inc)

# Output file name
ds_save_p = project_dir.joinpath("data", "tmp", key)

# Mask to restrict the ds to all included ROIs
mask = mask_dir.joinpath("all_ROIs.nii.gz")

# Targets to include
targets = ['FORREST', 'MRS. GUMP', 'FORREST (V.O.)', 'LT. DAN']

# Prep ROIs
roi_info = uf.prepare_roi_info(roi_masks_list)

# Preprocess each run for each participant individually
ds = uf.load_create_save_ds(ds_save_p, data_files, ref, warp_files, mask, 
                            detrending="polynomial", use_zscore=True, use_events=True, 
                            anno_dir=anno_dir, use_glm_estimates=False, targets=targets, 
                            event_offset=0, event_dur=8, rois=roi_info, save_disc_space=True)

# print the used mask
print "Mask used: \n{}\n".format(mask)

# print the shape of the ds
print "Shape of the dataset: {}\n".format(ds.shape)

# print roi info
rois = [roi for roi in ds.fa if "ROI" in roi]
for roi in rois:
    sub_ds = ds[:, {roi: [1]}]
    print "Number of voxels included in the {}: {}".format(roi, sub_ds.shape[1])
print "\n"

# print event info 
all_events = ds.sa.targets
unique_events = np.unique(all_events)
num_events = len(ds.sa.targets)
print "Included targets: {}".format(unique_events)
print "Number of included events: {}\n".format(num_events)

print ds.a
print ds.fa
print ds.sa


# In[ ]:




