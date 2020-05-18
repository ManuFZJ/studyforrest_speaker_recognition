#!/usr/bin/env python
# coding: utf-8

# infile = sys.argv[1]
# annotated_time = sys.argv[2]
# target_time = sys.argv[3]
# outdir = sys.argv[4]

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
mask_dir = project_dir.joinpath("data", "tmp", "masks")
roi_mask_dir = project_dir.joinpath("data", "tmp", "masks", "roi_masks")

# Get Participant Files
ao_files = uf.get_files(data_dir, '*aomovie*nii.gz')
av_files = uf.get_files(data_dir, '*avmovie*nii.gz')

# Get ROI Files
roi_masks_list = uf.get_files(roi_mask_dir, '*.nii.gz')

# Get Annotation Files
new_anno_dir = project_dir.joinpath("data", "tmp", "speech_anno")
ao_anno_files = uf.get_files(new_anno_dir.joinpath("aomovie"), '*.tsv')
av_anno_files = uf.get_files(new_anno_dir.joinpath("avmovie"), '*.tsv')

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


# In[11]:


# DS
ds_save_p = project_dir.joinpath("data", "tmp","preprocessed_ds.hdf5")

# Maske schließt noch zu viel ein (combi aus allen roi masken besser)
mask = project_dir.joinpath("data", "tmp", "masks", "gm_mni_nilearn_mask_adjusted.nii.gz")

targets = ['FORREST', 'MRS. GUMP', 'FORREST (V.O.)', 'LT. DAN']
roi_info = uf.prepare_roi_info(roi_masks_list)

# each run for each participant will be preprocessed individually
ds = uf.load_create_save_ds(ds_save_p, av_files[0:16], ref_space=ref, warp_files=warp_files,
                            mask=mask, detrending="polynomial", use_zscore=True, use_events=True,
                            anno_dir=new_anno_dir, use_glm_estimates=False, targets=targets,
                            event_offset=0, event_dur=3, rois=roi_info)

print "Shape of the dataset: {}\n".format(ds.shape)

print "Mask used: \n{}\n".format(mask)

all_events = ds.sa.targets
unique_events = np.unique(all_events)
num_events = len(ds.sa.targets)

print "Included unique events: {}".format(unique_events)
print "Number of included events: {}\n".format(num_events)

print ds.a
print ds.fa
print ds.sa

# verwendete Zeiteinheit nach event detection: ms

# noch zu viele voxel eingeschlossen
    # später wird als maske der Zusammenschluss aller roi masken verwendet


# In[6]:


# Distribution of speaker events over the segments
num_ev_df = uf.num_same_event(ds, targets)

num_ev_plot = num_ev_df.plot(kind="bar", width=0.8, figsize=[12,8])
num_ev_plot.set_xlabel("Segments", labelpad=20, weight='bold', size=12)
num_ev_plot.set_ylabel("Number of events", labelpad=20, weight='bold', size=12)


# In[14]:


# Pauses between sentences of the same speaker
print "Pauses between sentences of the same speaker"
within_speaker = uf.breaks_speaker(ao_anno_files, targets, True)

sns.set(color_codes=True)
f, axes = plt.subplots(8, 4, figsize=(20, 25), sharex=True)

for df, top_num in zip(within_speaker, range(0,len(within_speaker))):
    for target, sub_num in zip(targets, range(0,4)):
        sns.distplot(df.loc[:, target], color="k", kde=False, 
                     rug=True, ax=axes[top_num, sub_num])


# In[15]:


# Pauses between sentences of all speaker combinations
print "Pauses between sentences of different speakers"                     
between_speaker = uf.breaks_speaker(ao_anno_files, targets, False)

sns.set(color_codes=True)
f, axes = plt.subplots(8, 12, figsize=(25, 30), sharex=True)

for df, top_num in zip(between_speaker, range(0,len(between_speaker))):
    for pri_tar in targets:
        for sec_tar in targets:
            if sec_tar != pri_tar:
                column = "{} - {}".format(pri_tar, sec_tar)
                sns.distplot(df.loc[:, column], color="k", kde=False, rug=True, 
                             ax=axes[top_num, df.columns.get_loc(column)])


# In[63]:


# Use cubehelix to get a custom sequential palette
pal = sns.cubehelix_palette(8, rot=-.5, dark=.3)

# Show each distribution with both violins and points
fig, ax = plt.subplots(figsize=(25,15))

test_data = between_speaker[0].loc[:, ["FORREST - FORREST (V.O.)",
                                       "FORREST - MRS. GUMP", 
                                       "FORREST - LT. DAN",
                                       "FORREST (V.O.) - FORREST",
                                       "FORREST (V.O.) - MRS. GUMP",
                                       "FORREST (V.O.) - LT. DAN",
                                       "MRS. GUMP - FORREST",
                                       "MRS. GUMP - FORREST (V.O.)",
                                       "MRS. GUMP - LT. DAN"]]
sns.violinplot(data=test_data, ax=ax, palette=pal, inner="points", width=1, linewidth=1.2)


# In[14]:


# Normal Classifier Analysis
result_list = []
roi_names = ["IPL","aSTS","mSTS","pSTS", "V1"]

clf = mvpa.kNN(k=1, dfx=mvpa.one_minus_correlation, voting='majority')

for participant in np.unique(ds.sa.participant):
    participant_dict = {}
    
    for roi_name in roi_names:
        clf = mvpa.kNN(k=1, dfx=mvpa.one_minus_correlation, voting='majority')
        cvte = mvpa.CrossValidation(clf, mvpa.NFoldPartitioner(cvtype=1),
                                    errorfx=lambda p, t: np.mean(p == t),
                                    enable_ca=['stats']) 
        results = cvte(ds[{'participant': [participant]}, 
                          {roi_name: [1]}])
        
        key = "Participant {} - {}".format(participant, roi_name)
        participant_dict[key] = [cvte.ca.stats.as_string(description=True), 
                                 cvte.ca.stats.matrix]
        result_list.append(participant_dict)
        
# print result_list

# mit künstlichen Daten die einzelnen splits ausgeben lassen
# mit: cvtype=1, ohne count und selection_strat arbeiten
# count=10, selection_strategy="random", attr="participant"


# In[17]:


# RSA
## Graphical Results
def plot_mtx(mtx, labels, title):
    pl.figure()
    pl.imshow(mtx, interpolation='nearest')
    pl.xticks(range(len(mtx)), labels, rotation=-45)
    pl.yticks(range(len(mtx)), labels)
    pl.title(title)
    pl.clim((0, 2))
    pl.colorbar()

roi_datatsets = [ds[:, ds.fa.IPL != 0],
                 ds[:, ds.fa.aSTS != 0],
                 ds[:, ds.fa.mSTS != 0],
                 ds[:, ds.fa.pSTS != 0],
                 ds[:, ds.fa.V1 != 0]]
roi_names = ["IPL","aSTS","mSTS","pSTS", "V1"]

for roi_ds, roi_name in zip(roi_datatsets, roi_names):
    mtgs = mvpa.mean_group_sample(['targets'])
    mtds = mtgs(roi_ds)
    dsm = rsa.PDist(square=True)
    res = dsm(mtds)

    plot_mtx(res, mtds.sa.targets, '{}-ROI pattern correlation distances'.format(roi_name))
    pl.show()


# In[ ]:


# 18.5
    # need swig to install pymvpa2
    # 


# In[ ]:


# Notizen letzte Conference
    # wenn maske binär dann threshold reporten
    # vllt viele Modelle erstellen und gucken wie stark sie sich unterscheiden bzw.
    # was für welche Unterschiede verantwortlich ist

# juseless: man condor_submit

# wie führe ich die analyse auf dem cluster aus?
    # https://docs.inm7.de/tools/htcondor/ 
    # https://docs.inm7.de/cluster/htcondor/


# lieber ohne event offset wenn kein averaging
# offset wenn mit averaging
# so eher für block design
# gucken ob stimulus block design mäßig ist


# Analyse
    # selbe anzahl von zufälligen samples für die einzelnen identities ziehen
    # für training set zb 100 pro identity
    # prediction für was übrig ist

# Classifier Code
    # http://www.pymvpa.org/examples/pylab_2d.html


# In[ ]:


## Mask War
    # Neuer Ansatz mit grey matter mask
        # Erstellprozess: resampled MNI Grey Matter mask to grp_space, made values binary
        # Überschneidung zwischen binary resampled gm mask und dilatated roi mask

        # Sonderfall anterior temporal mask --> Überschneidung mit sts mask --> überschneidung vom Ergebnis
        # mit gm mask

        # dilation verwenden!
        # dilatation --> graue substanz maske --> roi maske

    # noch zu viele unpassende Datenpunkte enthalten in den roi masks
    # rumspielen und logical_not verwenden? um gut begrenzte ROI-masks zu generieren
        # logical_or überschreibt false mit true? --> damit maske mit allen rois erstellen
            # dient dann als standard mask für ds
    
ori_mask = "/home/arkani/Desktop/studyforrest_speaker_recognition/data/tmp/masks_tmp/ipl_association-test_z_FDR_0.01_adjusted.nii.gz"
overlap_mask = "/home/arkani/Desktop/studyforrest_speaker_recognition/data/tmp/masks_tmp/gm_mni_nilearn_mask.nii.gz"

new_mask = uf.get_adjusted_mask(ori_mask, ref) # , overlap_mask=overlap_mask


# In[ ]:


# Prep Speech Anno used cmd commands
    # python researchcut2segments.py fg_rscut_ad_ger_speech_tagged.tsv "avmovie" "avmovie" .
    # python researchcut2segments.py fg_rscut_ad_ger_speech_tagged.tsv "aomovie" "aomovie" .

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

