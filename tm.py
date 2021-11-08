import pandas as pd
import numpy as np
import nibabel as nib
import time
import cy_getRpthreaded
import psutil
import plotly.graph_objects as go
import roi_vis
import os

def scannerspace_from_index(A, i, j ,k):
    """

    :param A: affine transformation matrix inherent as a property of the image
    :param i: x coord voxel index
    :param j: y coord voxel index
    :param k: z coord voxel index
    :return: triple (x, y, z) in mm space
    """
    M = A[:3, :3]
    abc = A[:3, 3]
    return M.dot([i, j, k]) + abc

def dices(binarizedTopPerc, networkVolumes):
    """
    Compute all the regions dice coefficients.
    :param networkVolumes: all previously generated masks for each network. shape=(13, 91, 109, 91), dtype=np.bool_
                           networkVolumes[i] returns the binary volume of network i, where True indicates proximity
                           to the provided ROIs given a priori in 153ProbabilisticROIs_MNI_info.txt.
    :param binarizedTopPerc: the seed voxel binary volume, shape=(91, 109, 91), dtype=np.bool_ where True
                             indicates the voxel is within the top percent of functionally correlated voxels
                             with the seed voxel
    :return: dices, ndarray of 13 floats, where dices[i] is the ith network degree of overlap. index mapping
             in index2netname.csv
    """
    dices = np.zeros(networkVolumes.shape[0])
    for i in range(networkVolumes.shape[0]):
        binaryNetworkVol = networkVolumes[i]
        AND = np.bitwise_and(binarizedTopPerc, binaryNetworkVol)
        ANDsize = np.count_nonzero(AND)
        dices[i] = ANDsize
    return dices

absolute_start = time.time()

# Following procedure outlined in Methods, Section 2.3.1, of DOI: 10.1016/j.neuroimage.2021.118164
# Load required params
params = pd.read_csv('parameters.csv', header=None, index_col=0, squeeze=True).to_dict()
BOLD_time = nib.load(params["BOLD_path"]).get_fdata()
MASK = nib.load(params["MASK_path"]).get_fdata()
local_rm_thresh = float(params["local_rm_thresh"])
dice_top_perc = float(params["dice_top_perc"])

# AFFINE transformation matrix, verified visually, converts voxels to scanner (MNI) space.
AFFINE = np.array([[2.63764, 0, 0, -121],
                   [0, 2.19883, 0.10776, -126],
                   [0, -0.11941, 1.9844, -72],
                   [0, 0, 0, 1]])

# Load ROI data
dfMNI = pd.read_csv("153ProbabilisticROIs_MNI_info.txt", delimiter="\t")
index2net = pd.read_csv('index2netname.csv', header=None, index_col=0, squeeze=True).to_dict()
n_networks = len(index2net)
# This tupled list only needed for next line, to ensure ordering.
ordered_networks = [(i, index2net[i]) for i in range(n_networks)]
# Dictionary that maps a network index to a list of ROIs (each roi is a triple in MNI space)
index2rois = {i:np.array([[dfMNI["x"][j], dfMNI["y"][j], dfMNI["z"][j]]
                          for j in np.argwhere(np.array(dfMNI["Net. Name"]) == net).flatten()])
              for i,net in ordered_networks}

# m = number of gray matter voxels masked
m = len(np.argwhere(MASK == 1.0))
print("Size of mask before filtering 0 time courses:  " + str(m) + " voxels")

# Loading in network volumes from the disk
networkVolumes = np.zeros(shape=(n_networks, 91, 109, 91), dtype=np.bool_)
template_mask_files = sorted(os.listdir(os.getcwd() + "/BinaryTemplateMasks"), key=lambda x: int(x[:2]) if x[1] != '-' else int(x[0]))
print("Network binary mask files: ")
print(*template_mask_files, sep="\n")
for i,filename in enumerate((template_mask_files)):
    with open("BinaryTemplateMasks/" + filename, 'rb') as f:
        vol = np.load(f)
    networkVolumes[i] = vol

# Iterate over the fourth dimension of time, generating the column wise matrix needed for correlation
# T - each row is a gray matter voxel time series of BOLD response
T = np.zeros(shape=(m, BOLD_time.shape[3]), dtype=np.float64)
# Map the row index to its MNI location
row2MNI = np.zeros(shape=(m, 3), dtype=np.single)
# Map the row index to its Voxel location
row2voxel = np.zeros(shape=(m, 3), dtype=np.int32)

# Populate T, row2MNI, row2Voxel
for n, triple in enumerate((np.argwhere(MASK == 1.0))):
    i, j, k = triple
    row = BOLD_time[i, j, k, :]
    T[n, :] = row
    row2MNI[n, :] = scannerspace_from_index(AFFINE, i, j, k)
    row2voxel[n, :] = i, j, k

# Filter rows containing all zeros
row2MNI = row2MNI[~np.all(T == 0.0, axis=1)]
row2voxel = row2voxel[~np.all(T == 0.0, axis=1)]
T = T[~np.all(T == 0.0, axis=1)]

nzero = T.shape[0]
print("Non-zero total gray matter voxels:  " + str(nzero))

# R = Functional Connectivity Matrix. Allocate its memory, pass to C++ multithreading.
R = np.zeros(shape=(nzero, nzero), dtype=np.single)

# ------- DO NOT MODIFY! -----------  The C++ can be very rigid, no change to this function is recommended.
# Correctness is verified. The correlation matrix is correct.
start = time.time()
R = cy_getRpthreaded.getRpthreaded_py(T, R, nzero, BOLD_time.shape[-1])
print("C++ multithreading finished in:  " + str(time.time() - start) + " seconds")

# Assigning structure as volume, -1 defaults to no assignment. Set all to -1 to begin
voxel_assignVOL = np.zeros(shape=BOLD_time.shape[:3], dtype=np.int32) - 1
top_amount = int(nzero * dice_top_perc)

# Iterate through each column of R, argsort, cut the top percentage, then generate the binary mask for this gmv (gray matter voxel) seed
binarizedTopPerc = np.zeros(shape=BOLD_time.shape[:3], dtype=np.bool_)
for gmv in range(nzero):
    all_correlates = R[:, gmv]
    top5perc = np.argsort(all_correlates)[::-1][:top_amount]
    binarizedTopPerc.fill(False)
    for index in top5perc:
        if np.linalg.norm(row2MNI[index] - row2MNI[gmv]) <= local_rm_thresh: continue
        i, j, k = row2voxel[index]
        binarizedTopPerc[i, j, k] = True


    dcoefs = dices(binarizedTopPerc, networkVolumes)

    if sum(dcoefs) == 0.0:
        print("All zero dice coefficients were found")
        continue

    # Get the largest dice coefficient, assign it
    assignment = np.argmax(dcoefs)
    ai, aj, ak = row2voxel[gmv]
    voxel_assignVOL[ai, aj, ak] = assignment


print("Unique voxels in volume: " + str(np.unique(voxel_assignVOL)))

# TODO clusters of less than 4 continuous voxels - remove from network map
network_assigned = nib.Nifti1Image(voxel_assignVOL, AFFINE, nib.Nifti1Header())
nib.save(network_assigned, params["output_file"])
print("done!")
print("Entire job finished in  " + str(time.time() - absolute_start) + " seconds")
