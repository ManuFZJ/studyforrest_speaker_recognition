#!/usr/bin/env python
# coding: utf-8

# In[2]:


# If an import error occurs try: 
# "pip install attrs==19.1.0" in the cmd
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

# Get Participant Files
ao_files = uf.get_files(data_dir, '*aomovie*nii.gz')
av_files = uf.get_files(data_dir, '*avmovie*nii.gz')

# Get Annotation Files
ao_anno_files = uf.get_files(anno_dir, '*emotions*ao*run*.tsv')
av_anno_files = uf.get_files(anno_dir, '*emotions*av*run*.tsv')

# Reference Space
ref = str(template_dir.joinpath("templates", "grpbold3Tp2", "brain.nii.gz"))

# Warp Files
warp_files = uf.get_files(template_dir, '*subj2tmpl_warp*.nii.gz')

# Check if all components are good to go
path_list = [project_dir, data_dir, anno_dir, template_dir, ref]
file_lists = [ao_files, av_files, ao_anno_files, av_anno_files, warp_files]

check = uf.check_all_components(path_list, file_lists)
print("{}\n".format(check[0]))
print("{}\n".format(check[1]))


# In[6]:


# Create/Load DS
ds_save_p = Path("/home/arkani/Desktop/studyforrest_speaker_recognition/data/tmp/preprocessed_ds.hdf5")
mask = str(project_dir.joinpath("data", "speakers_association-test_z_FDR_0.01.nii.gz"))
# targets = ['FORREST', 'MRSGUMP', 'FORRESTVO', 'FORRESTJR', 'BUBBA', 'DAN', 'JENNY']
targets = ['FORREST', 'MRSGUMP', 'FORRESTVO', 'JENNY']

ds = uf.load_create_save_ds(ds_save_p, av_files[0:2], ref_space=ref, warp_files=warp_files,
                            mask=mask, detrending=True, use_zscore=True, use_events=True,
                            anno_dir=anno_dir, use_glm_estimates=False, targets=targets,
                            event_offset=2, event_dur=6)

print("This is the shape of the dataset that will be used in the following Analysis: {}\n".format(ds.shape))

print("This mask file was used and if required resampled: \n{}\n".format(mask))

all_events = ds.sa.targets
unique_events = np.unique(all_events)
num_events = len(ds.sa.targets)

print("The following unique events will be included: {}".format(unique_events))
print("The number of included events amounts to: {}\n".format(num_events))

print(ds.a)
print(ds.fa)
print(ds.sa)


# In[7]:


# Normal Classifier Analysis
clf = mvpa.kNN(k=1, dfx=mvpa.one_minus_correlation, voting='majority')
clf.train(ds)
predictions = clf.predict(ds.samples)
print(np.mean(predictions == ds.sa.targets))


# In[8]:


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


# In[9]:


# RSA
mtgs = mvpa.mean_group_sample(['targets'])
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

plot_mtx(res, mtds.sa.targets, 'ROI pattern correlation distances')
pl.show()


# In[9]:


## MVPA (GLM Estimates)
    # useful when: multiple concurrent signals are overlapping in time, such as in 
    # fast event-related designs
    # GLM estimates instead of raw data

# only 1 ds as input possible otherwise unstackable
# modify event dicts and stack datasets before event detection to make this work?
    
# Create Datset
mask_files = uf.get_files(project_dir, '*association*.nii.gz')
mask = mask_files[0]
# targets = ['FORREST', 'MRSGUMP', 'FORRESTVO', 'FORRESTJR', 'BUBBA', 'DAN', 'JENNY']
# vstack Probleme wenn nicht in allen Runs die gleichen Targets gefunden werden?
targets = ['FORREST', 'MRSGUMP', 'FORRESTVO', 'FORRESTJR', 'DAN', 'JENNY']
ds = uf.preprocess_datasets(av_files[0], ref_space=ref, warp_files=warp_files, 
                            mask=mask, detrending=True, use_zscore=True, 
                            use_events=True, anno_dir=anno_dir, use_glm_estimates=True,
                            targets=targets, event_offset=2, event_dur=6)

print("This is the Dataset that will be used in the following Analysis:")
print("{}\n".format(ds))

print("Used mask file: \n{}\n".format(mask))
print(ds.sa)
print(ds.fa)
print(ds.a)

print("\nGLM Estimates were computed for the following targets: {}\n".format(ds.sa.targets))

# cross-validation Analysis with a chosen Classifier
clf = mvpa.kNN(k=1, dfx=mvpa.one_minus_correlation, voting='majority')
cv = mvpa.CrossValidation(clf, mvpa.NFoldPartitioner(attr='chunks'))
# cv_glm = cv(ds)
# print('%.2f' % np.mean(cv_glm))


# In[ ]:


# kürzere code lines (pylint)

# vstack Probleme
    # glm estimates erzeugen Problem
    
# mask
    # restrict mask to ROIs
    # nilearn Methode: another Mask, Überschneidung, dann Dilatation
    # wie Info aus Atlanten verwenden? 
        # ROI representations in xml files (Bsp.: "/usr/share/data/mni-structural-atlas/MNI.xml")
            # Definition des ROIs für MNI 1mm
            # wie ROI Info anpassen an Grp Space

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
    
# Vorgehensweise
    # jeden einzelnen Run mit dem jeweiligen Template für die VP in den Gruppenspace
        # sind momentan alle nur innerhalb der VP aligned
    # Maske an Gruppenraum anpassen
    # Maske bearbeiten (nur Sprachareale)

