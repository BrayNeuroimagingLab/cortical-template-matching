# TemplateMatching

*Author: Rylan Marianchuk*
*August 2021*

Python package dependences:
```
numpy
nibabel
pandas
cython
```

Or if using on the high performance cluster in ```bray_bulk``` activate the miniconda containing these dependencies pre-installed,

```
export PATH=/bulk/bray_bulk/software/miniconda3/bin:$PATH
source activate TemplateMatch
```

Navigate to ```/bulk/bray_bulk/cortical-template-matching``` on ARC.

Before executing the template matching procedure as described in Methods, Section 2.3.1, of DOI: 10.1016/j.neuroimage.2021.118164,
populate ```parameters.csv```:

```BOLD_path``` the path to the .nii neuroimage of BOLD activations across time (the scan)

```MASK_path``` the path to the .nii binary mask of the subject with same shape as image scan.

```local_rm_thresh``` the distance threshold in scanner space that will be removed from correlates in the matching. i.e. if set to 20, any gray matter voxel less than 20mm away from the computing voxel is ommitted from its dice overlap.

```dice_top_perc``` percentage of all other correlated gray matter voxels to consider in the dice coefficient overlap. Used to filter those correlations by change.

```output_file``` the string to name the assigned template (output).

Once these are set properly, execute

```
sbatch run.sh
```

which calls the main python file ```tm.py``` (tm = template match). Jobs usually take an hour to run.

