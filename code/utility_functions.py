from pathlib import Path
import re
import os.path
import time
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

# Get participant + run number + stimulus type
def find_participant_info(file_string):
    all_nums = re.findall("\d+", file_string)
    part_num = all_nums[-2]
    run_num = all_nums[-1]
    movie_type = file_string.find("ao")
    if movie_type == -1:
        movie_type = "av"
    else:
        movie_type = "ao"
    return [int(part_num), int(run_num), movie_type]

# Create event dict for the specific run of the individual dataset
def create_event_dict(anno_files_dir, ds_path, anno_type, columns, targets):
    part_info = find_participant_info(ds_path)
    correct_anno_file = get_files(anno_files_dir, "*emotions*{}*run*{}*.tsv".format(anno_type,
                                                                                    part_info[1]))
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
def get_adjusted_mask(ori_mask, ref_space):
    new_mask_p = ori_mask[:-7] + "_adjusted.nii.gz"
    if os.path.isfile(new_mask_p):
        new_mask = nib.load(new_mask_p)
    else:
        mask = nib.load(ori_mask)
        ds = mvpa.fmri_dataset(samples=ref_space)
        affine = ds.a.imgaffine
        shape = ds.a.voxel_dim
        new_mask = nil.image.resample_img(mask, target_affine=affine,
                                          target_shape=shape, clip=True,
                                          interpolation="continuous")
        new_mask_data = new_mask.get_fdata()
        bool_data = new_mask_data.astype("int32") # astype(bool) testen?
        new_mask = nil.image.new_img_like(new_mask, bool_data, affine=new_mask.affine)
        nib.save(new_mask, "{}_adjusted.nii.gz".format(ori_mask[:-7]))
    return new_mask

# Preprocess individual dataset
def preprocessing(ds_p, ref_space, warp_files, mask_p, detrending, use_zscore, use_events,
                  anno_dir, use_glm_estimates, targets, event_offset, event_dur):
   
    vp_num_str = ds_p[(ds_p.find("sub")+4):(ds_p.find("sub")+6)]
    warp_file = [file for file in warp_files if file.find(vp_num_str) != -1][0]
    
    part_info = [ds_p[(ds_p.find("sub")+4):(ds_p.find("sub")+6)], 
                        ds_p[(ds_p.find("task")+5):(ds_p.find("task")+7)],
                        ds_p[(ds_p.find("run")+4):(ds_p.find("run")+5)]
                        ]
    temp_file = str((Path.cwd().parents[0]).joinpath("data", "tmp", 
                            "sub-{}_{}-movie_run-{}_warped_file.nii.gz".format(part_info[0],
                            part_info[1], part_info[2])))
    warped_ds = warp_image(ds_p, ref_space, warp_file, temp_file)

    while not os.path.exists(warped_ds):
        time.sleep(5)

    if os.path.isfile(warped_ds):
        if mask_p != None:
            mask = get_adjusted_mask(mask_p, ref_space)
            ds = mvpa.fmri_dataset(samples=warped_ds, mask=mask)
        else:
            ds = mvpa.fmri_dataset(samples=warped_ds)
        
    part_info = find_participant_info(ds_p)
    ds.sa['participant'] = [part_info[0] * ds.shape[0]]
    ds.sa['chunks'] = [part_info[1] * ds.shape[0]]
    if detrending == True:
        detrender = mvpa.PolyDetrendMapper(polyord=1)
        ds = ds.get_mapped(detrender)
    if use_zscore == True:
        mvpa.zscore(ds)
    if use_events == True:
        events = create_event_dict(anno_dir, ds_p, part_info[2], 
                        ['onset', 'duration', 'targets'], targets)
        if use_glm_estimates == True:
            ds = mvpa.fit_event_hrf_model(ds, events, time_attr='time_coords',
                        condition_attr='targets')
        else:
            ds = mvpa.extract_boxcar_event_samples(ds, events=events,
                        time_attr='time_coords', match='closest', event_offset=event_offset, 
                        event_duration=event_dur,  eprefix='event', event_mapper=None)
    return ds

# Preprocess multiple datasets
def preprocess_datasets(dataset_list, ref_space, warp_files, mask, detrending, use_zscore,
                        use_events, anno_dir, use_glm_estimates, targets, event_offset, event_dur):
    if isinstance(dataset_list, list):
        datasets = [preprocessing(ds_p, ref_space, warp_files, mask, detrending, use_zscore,
                                  use_events, anno_dir, use_glm_estimates, targets,  event_offset,
                                  event_dur)
                            for ds_p in dataset_list]
        ds = mvpa.vstack(datasets, a='drop_nonunique', fa='drop_nonunique')
    else:
        ds = preprocessing(dataset_list, ref_space, warp_files, mask, detrending,
                                        use_zscore, use_events, anno_dir, use_glm_estimates,
                                        targets, event_offset, event_dur)
    return ds
    