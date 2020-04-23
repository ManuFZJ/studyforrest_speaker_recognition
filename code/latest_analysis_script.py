#!/usr/bin/env python
# coding: utf-8

# In[2]:


# If an import error occurs try:
# pip install attrs==19.1.0
# in the cmd
from pathlib import Path
import numpy as np
import pylab as pl
from numpy.testing.decorators import skipif
import mvpa2.suite as mvpa
from mvpa2.measures import rsa
import utility_functions as uf


# In[3]:


# Wider Output
np.set_printoptions(edgeitems=20)

# Project Directory
project_dir = Path.cwd().parents[0]

# BOLD Files Directory
data_dir = project_dir.joinpath("data", "studyforrest-data-aligned")

# Anno Files Directory
anno_dir = project_dir.joinpath("data", "studyforrest-data-annotations")

# Template Directory
template_dir = project_dir.joinpath("data", "studyforrest-data-templatetransforms")

# Mask Directory
mask_dir = project_dir.joinpath("data", "masks")
roi_mask_dir = project_dir.joinpath("data", "masks", "roi_masks")

# Get Participant Files
ao_files = uf.get_files(data_dir, '*aomovie*nii.gz')
av_files = uf.get_files(data_dir, '*avmovie*nii.gz')

# Get ROI Files
roi_masks_list = uf.get_files(roi_mask_dir, '*.nii.gz')

# Get Annotation Files
new_anno_dir = project_dir.joinpath("data","tmp","speech_anno")
ao_anno_files = uf.get_files(new_anno_dir.joinpath("aomovie"), '*.tsv')
av_anno_files = uf.get_files(new_anno_dir.joinpath("avmovie"), '*.tsv')

#/home/arkani/Desktop/studyforrest_speaker_recognition/data/tmp/speech_anno/aomovie

# Reference Space
ref = template_dir.joinpath("templates", "grpbold3Tp2", "brain.nii.gz")

# Warp Files
warp_files = uf.get_files(template_dir, '*subj2tmpl_warp*.nii.gz')

# Check if all components are good to go
path_list = [project_dir, data_dir, anno_dir, template_dir, ref]
file_lists = [ao_files, av_files, ao_anno_files, av_anno_files, warp_files, roi_masks_list]

check = uf.check_all_components(path_list, file_lists)
print "{}\n".format(check[0])
print "{}\n".format(check[1])


# In[7]:


# DS
ds_save_p = project_dir.joinpath("data","tmp","preprocessed_ds.hdf5")

# Maske schließt noch zu viel ein (combi aus allen roi masken besser)
mask = project_dir.joinpath("data","masks","gm_mni_nilearn_mask_adjusted.nii.gz")

# targets = ['FORREST', 'MRSGUMP', 'FORRESTVO', 'JENNY', 'FORRESTJR', 'BUBBA', 'DAN']
targets = ['FORREST', 'MRS. GUMP', 'FORREST (V.O.)', 'LT. DAN']
roi_info = uf.prepare_roi_info(roi_masks_list)

# each run for each participant will be preprocessed individually
ds = uf.load_create_save_ds(ds_save_p, av_files[0:16], ref_space=ref, warp_files=warp_files,
                            mask=mask, detrending="polynomial", use_zscore=True, use_events=True,
                            anno_dir=new_anno_dir, use_glm_estimates=False, targets=targets,
                            event_offset=0, event_dur=4, rois=roi_info)

print "Shape of the dataset: {}\n".format(ds.shape)

print "Mask used: \n{}\n".format(mask)

all_events = ds.sa.person
unique_events = np.unique(all_events)
num_events = len(ds.sa.person) # 5048

print "Included unique events: {}".format(unique_events)
print "Number of included events: {}\n".format(num_events)

print ds.a
print ds.fa
print ds.sa

# verwendete Zeiteinheit nach event detection: ms


# In[ ]:


# wie führe ich die analyse auf dem cluster aus?
    # HTCondor Infos durchlesen
        # https://docs.inm7.de/tools/htcondor/ 
        # https://docs.inm7.de/cluster/htcondor/ 
    # mit minimal script ausprobieren


# lieber ohne event offset wenn kein averaging
# offset wenn mit averaging
## so eher für block design
## gucken ob stimulus block design mäßig ist


# Analyse
    # Forrest, Forrest VO, Dan, Jenny
    # selbe anzahl von zufälligen samples für die einzelnen identities ziehen (aus allen einzelnen runs?)
    # für training set zb 100 pro identity
    # prediction für was übrig ist


# In[9]:


# Prep Speech Anno
# used cmd commands
    # python researchcut2segments.py fg_rscut_ad_ger_speech_tagged.tsv "avmovie" "avmovie" .
    # python researchcut2segments.py fg_rscut_ad_ger_speech_tagged.tsv "aomovie" "aomovie" .

# speechanno event selection
    # erstes event nach speaker change
    # duration wird von boxcar func auf 3 gesetzt


# In[ ]:


## Mask War
    # Neuer Ansatz mit grey matter mask
        # Erstellprozess: resampled MNI Grey Matter mask to grp_space, made values binary
        # Überschneidung zwischen binary resampled gm mask und dilatated roi mask

        # Sonderfall anterior temporal mask --> Überschneidung mit sts mask --> überschneidung vom Ergebnis
        # mit gm mask

        # dilation verwenden!
        # dilatation --> graue substanz maske --> roi maske

        # auch bei Überschneidung etc die daten binär machen !!!
        # alle masken namen eindeutig und klar machen
        # alle verwendeten masken binär machen

    # noch zu viele unpassende Datenpunkte enthalten in den roi masks
    # rumspielen und logical_not verwenden? um gut begrenzte ROI-masks zu generieren
        # logical_or überschreibt false mit true? --> damit maske mit allen rois erstellen
            # dient dann als standard mask für ds
    
ori_mask = "/home/arkani/Desktop/studyforrest_speaker_recognition/data/tmp/masks_tmp/ipl_association-test_z_FDR_0.01_adjusted.nii.gz"
ref_space = ref
overlap_mask = "/home/arkani/Desktop/studyforrest_speaker_recognition/data/tmp/masks_tmp/gm_mni_nilearn_mask.nii.gz"


new_mask = uf.get_adjusted_mask(ori_mask, ref_space) # , overlap_mask=overlap_mask


# In[12]:


print(ds[:, ds.fa.psts != 0].shape)
print(ds[:, ds.fa.mtg != 0].shape)


# In[8]:


# RSA
mtgs = mvpa.mean_group_sample(['person'])
mtds = mtgs(ds)
dsm = rsa.PDist(square=True)
res = dsm(mtds)

## Graphical Results
def plot_mtx(mtx, labels, title):
    pl.figure()
    pl.imshow(mtx, interpolation='nearest')
    pl.xticks(range(len(mtx)), labels, rotation=-45)
    pl.yticks(range(len(mtx)), labels)
    pl.title(title)
    pl.clim((0, 2))
    pl.colorbar()

plot_mtx(res, mtds.sa.person, 'ROI pattern correlation distances')
pl.show()


# In[9]:


# Normal Classifier Analysis
clf = mvpa.kNN(k=1, dfx=mvpa.one_minus_correlation, voting='majority')

#clf.train(ds_split1)

#predictions = clf.predict(ds_split2.samples)
#print(np.mean(predictions == ds_split2.sa.targets))


# In[8]:


# Testing
clf.set_postproc(mvpa.BinaryFxNode(mvpa.mean_mismatch_error, 'targets'))
clf.train(ds)
err = clf(ds)
print(err.samples)

clf = mvpa.kNN(k=1, dfx=mvpa.one_minus_correlation, voting='majority')
cvte = mvpa.CrossValidation(clf, mvpa.HalfPartitioner(attr='chunks'))
cv_results = cvte(ds)
print(np.mean(cv_results))

cvte = mvpa.CrossValidation(clf, mvpa.HalfPartitioner(attr='chunks'),
                       errorfx=lambda p, t: np.mean(p == t))
cv_results = cvte(ds)
print(np.mean(cv_results))
print(cv_results.samples)


cvte = mvpa.CrossValidation(clf, mvpa.NFoldPartitioner(cvtype=1), 
                            errorfx=lambda p, t: np.mean(p == t),
                            enable_ca=['stats'])
# cv_results = cvte(ds)
# print(cvte.ca.stats.as_string(description=True))
# print(cvte.ca.stats.matrix)


# In[ ]:


# kürzere code lines (pylint)

# h5f files abhängig von packages
    # lieber np arrays zum archivieren

# Tests
    # Regression Tests: 
        # 1. finaler Code: Ergebnis speichern, 
        # nach Code Änderungen gucken, ob das gleiche rauskommt
        # über mvpa.seed Zufallswert festlegen, sonst entstehen 
        # jedes mal andere Ergebnisse
    # Unit tests
        # gucken ob output in richtiger relation zu input
        # zb gleiche m x n konfusionsmatrix
    # Datenintegritätschecks
        # für argument x wirklich eine Liste / str angegeben

