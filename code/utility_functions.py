from pathlib import Path
import re
import os.path
import time
import random
import subprocess

import numpy as np
import pandas as pd
import nibabel as nib
import nilearn as nil

from numpy.testing.decorators import skipif
import mvpa2.suite as mvpa

# Get files
def get_files(dataset_dir, rglob_string):
    files = [str(Path("/").joinpath((Path(file).relative_to("/"))))
             for file in dataset_dir.rglob(rglob_string)]
    files = np.sort(files).tolist()
    return files

# Check if file/dir exists
def check_exi(path):
    path = Path(path)
    if path.exists():
        output = True
    else:
        output = False
    return output

# Check for file availability
def check_file_completeness(file_list):
    list_element = random.choice(file_list)
    if "ao" in list_element and len(file_list) == 112:
        output = True
    elif "av" in list_element and len(file_list) == 120:
        output = True
    elif "events" in list_element and len(file_list) == 8:
        output = True
    elif "warp" in list_element and len(file_list) == 15:
        output = True
    else:
        output = False
    return output

# Check if all components (directories, files, file_lists are complete)
def check_all_components(path_list, file_lists):
    path_exi_list = [
        check_exi(element)
        for element in path_list
    ]
    prob_paths_idx = [
        index
        for index, element in enumerate(path_exi_list)
        if element == False
    ]
    path_output = "All directories seem to be properly set up"
    if len(prob_paths_idx) != 0:
        path_output = [
            "This directory/file does not exist: {}".format(prob_ele)
            for prob_ele in path_list
            if path_list.index(prob_ele) in prob_paths_idx
        ]
    file_comp_list = [
        check_file_completeness(file_list)
        for file_list in file_lists
    ]
    prob_lists_idx = [
        index
        for index, element in enumerate(file_comp_list)
        if element == False
    ]
    lists_output = "All file lists seem to complete"
    if len(prob_lists_idx) != 0:
        lists_output = [
            "This list is incomplete: {}".format(prob_ele[0])
            for prob_ele in file_lists
            if file_lists.index(prob_ele) in prob_lists_idx
        ]
    return path_output, lists_output

# Get participant + run number + stimulus type
def find_participant_info(ds_p):
    ds_p = str(ds_p)
    part_info = [ds_p[(ds_p.find("sub") + 4):(ds_p.find("sub") + 6)],
                 ds_p[(ds_p.find("movie") - 2):(ds_p.find("movie"))],
                 ds_p[(ds_p.find("run") + 4):(ds_p.find("run") + 5)]
                 ]
    return part_info

# Create event dict for the specific run of the individual dataset
def create_event_dict(anno_files_dir, ds_path, anno_type, columns, targets):
    part_info = find_participant_info(ds_path)
    correct_anno_file = get_files(anno_files_dir, "*emotions*{}*run*{}*.tsv".format(anno_type,
                                                                                    part_info[2]))
    anno_info = pd.read_csv(correct_anno_file[0], delimiter="\t", header=0, names=columns,
                            usecols=columns)
    events = anno_info.loc[anno_info['targets'].isin(targets)]
    events_dict = events.to_dict(orient='records')  # records or index
    return events_dict

# Warp image to desired space
# change later so there is only one temp warped image
def warp_image(bold_file, ref_space, warp_file, output_path):
    if os.path.isfile(output_path) != True:
        subprocess.call(['applywarp', '-r', ref_space, "-i", bold_file, "-o",
                         output_path, "-w", warp_file])
    return output_path

# Find or generate fitting mask
def get_adjusted_mask(ori_mask, ref_space, **kwargs):
    overlap_mask = kwargs.get('overlap_mask', None)
    if overlap_mask is None:
        new_mask_p = ori_mask[:-7] + "_adjusted.nii.gz"
    else:
        new_mask_p = "{}_{}_overlap.nii.gz".format(ori_mask[:-7],
                                                   overlap_mask.split("/")[-1].split("_", 1)[0])
    if os.path.isfile(new_mask_p):
        new_mask = nib.load(new_mask_p)
    elif overlap_mask is not None and os.path.isfile(overlap_mask):
        mask = nib.load(ori_mask)
        overlap_mask = nib.load(overlap_mask)
        ds = mvpa.fmri_dataset(samples=ref_space)
        affine = ds.a.imgaffine
        shape = ds.a.voxel_dim
        mask_res = nil.image.resample_img(mask, target_affine=affine,
                                          target_shape=shape, clip=True,
                                          interpolation="continuous")
        overlap_mask_res = nil.image.resample_img(overlap_mask, target_affine=affine,
                                                  target_shape=shape, clip=True,
                                                  interpolation="continuous")
        mask_data = mask_res.get_fdata()
        overlap_mask_data = overlap_mask_res.get_fdata()
        in_common = np.logical_and(mask_data, overlap_mask_data).astype(np.int)
        new_mask = nil.image.new_img_like(mask, in_common, affine=mask.affine)
        nib.save(new_mask, new_mask_p)
    else:
        mask = nib.load(ori_mask)
        ds = mvpa.fmri_dataset(samples=ref_space)
        affine = ds.a.imgaffine
        shape = ds.a.voxel_dim
        new_mask = nil.image.resample_img(mask, target_affine=affine,
                                          target_shape=shape, clip=True,
                                          interpolation="continuous")
        new_mask_data = new_mask.get_fdata()
        bool_data = new_mask_data.astype("int32")  # astype(bool) testen?
        new_mask = nil.image.new_img_like(new_mask, bool_data, affine=new_mask.affine)
        nib.save(new_mask, "{}_adjusted.nii.gz".format(ori_mask[:-7]))
    return new_mask

# Preprocess individual dataset
def preprocessing(ds_p, ref_space, warp_files, mask_p, **kwargs):
    detrending = kwargs.get('detrending', True)
    use_zscore = kwargs.get('use_zscore', True)

    use_events = kwargs.get('use_events', False)
    anno_dir = kwargs.get('anno_dir', None)
    use_glm_estimates = kwargs.get('use_glm_estimates', False)
    targets = kwargs.get('targets', None)
    event_offset = kwargs.get('event_offset', None)
    event_dur = kwargs.get('event_dur', None)

    vp_num_str = ds_p[(ds_p.find("sub") + 4):(ds_p.find("sub") + 6)]
    warp_file = [file for file in warp_files if file.find(vp_num_str) != -1][0]
    part_info = find_participant_info(ds_p)
    temp_file_add = "sub-{}_{}-movie_run-{}_warped_file.nii.gz".format(part_info[0],
                                                                       part_info[1],
                                                                       int(part_info[2]))
    temp_file = str((Path.cwd().parents[0]).joinpath("data", "tmp", temp_file_add))
    warped_ds = warp_image(ds_p, ref_space, warp_file, temp_file)

    while not os.path.exists(warped_ds):
        time.sleep(5)

    if os.path.isfile(warped_ds):
        if mask_p != None:
            mask = get_adjusted_mask(mask_p, ref_space)
            ds = mvpa.fmri_dataset(samples=warped_ds, mask=mask)
        else:
            ds = mvpa.fmri_dataset(samples=warped_ds)

    ds.sa['participant'] = [int(part_info[0])]
    ds.sa['chunks'] = [int(part_info[2])]
    if detrending == True:
        detrender = mvpa.PolyDetrendMapper(polyord=1)
        ds = ds.get_mapped(detrender)
    if use_zscore == True:
        mvpa.zscore(ds)
    if use_events == True:
        events = create_event_dict(anno_dir, ds_p, part_info[1],
                                      ['onset', 'duration', 'targets'], targets)
        if use_glm_estimates == True:
            ds = mvpa.fit_event_hrf_model(ds, events, time_attr='time_coords',
                                          condition_attr='targets')
        else:
            ds = mvpa.extract_boxcar_event_samples(ds, events=events, time_attr='time_coords',
                                                   match='closest', event_offset=event_offset,
                                                   event_duration=event_dur, eprefix='event',
                                                   event_mapper=None)
    return ds

# Preprocess multiple datasets
def preprocess_datasets(dataset_list, ref_space, warp_files, mask, **kwargs):
    detrending = kwargs.get('detrending', True)
    use_zscore = kwargs.get('use_zscore', True)

    use_events = kwargs.get('use_events', False)
    anno_dir = kwargs.get('anno_dir', None)
    use_glm_estimates = kwargs.get('use_glm_estimates', False)
    targets = kwargs.get('targets', None)
    event_offset = kwargs.get('event_offset', None)
    event_dur = kwargs.get('event_dur', None)

    if isinstance(dataset_list, list):
        datasets = [preprocessing(ds_p, ref_space, warp_files, mask, detrending=detrending,
                                  use_zscore=use_zscore, use_events=use_events, anno_dir=anno_dir,
                                  use_glm_estimates=use_glm_estimates, targets=targets,
                                  event_offset=event_offset, event_dur=event_dur)
                    for ds_p in dataset_list]
        ds = mvpa.vstack(datasets, a='drop_nonunique', fa='drop_nonunique')
    else:
        ds = preprocessing(dataset_list, ref_space, warp_files, mask, detrending=detrending,
                           use_zscore=use_zscore, use_events=use_events, anno_dir=anno_dir,
                           use_glm_estimates=use_glm_estimates, targets=targets,
                           event_offset=event_offset, event_dur=event_dur)
    return ds

# Load or create and save ds
def load_create_save_ds(ds_save_p, dataset_list, ref_space, warp_files, mask, **kwargs):
    detrending = kwargs.get('detrending', True)
    use_zscore = kwargs.get('use_zscore', True)

    use_events = kwargs.get('use_events', False)
    anno_dir = kwargs.get('anno_dir', None)
    use_glm_estimates = kwargs.get('use_glm_estimates', False)
    targets = kwargs.get('targets', None)
    event_offset = kwargs.get('event_offset', None)
    event_dur = kwargs.get('event_dur', None)

    if ds_save_p.exists():
        ds = mvpa.h5load(str(ds_save_p))
    else:
        ds = preprocess_datasets(dataset_list, ref_space, warp_files, mask, detrending=detrending,
                                 use_zscore=use_zscore, use_events=use_events, anno_dir=anno_dir,
                                 use_glm_estimates=use_glm_estimates, targets=targets,
                                 event_offset=event_offset, event_dur=event_dur)
        mvpa.h5save(str(ds_save_p), ds)
    return ds
