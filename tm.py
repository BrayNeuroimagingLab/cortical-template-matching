import pandas as pd
import numpy as np
import nibabel as nib
import time
import getRpthreaded

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

def dice(triples1, triples2, diameter=7):
    """
    :return:
    """

    within_count = 0
    for trip1 in triples1:
        for trip2 in triples2:
            dist = np.linalg.norm(trip1 - trip2)
            if dist < diameter:
                within_count += 1
                # Can break because the ROI sheres are non-overlapping
                break
    return 2*within_count / (len(triples1) + len(triples2))

# Following procedure outlined in Methods, Section 2.3.1, of DOI: 10.1016/j.neuroimage.2021.118164
# Load required params
params = pd.read_csv('parameters.csv', header=None, index_col=0, squeeze=True).to_dict()
BOLD_time = nib.load(params["BOLD_path"]).getf_data()
MASK = nib.load(params["MASK_path"]).getf_data()
local_rm_thresh = float(params["local_rm_thresh"])
dice_top_perc = float(params["dice_top_perc"])
AFFINE = nib.load(params["BOLD_path"]).affine

# Load ROI data
dfMNI = pd.read_csv("153ProbabilisticROIs_MNI_info.txt", delimiter="\t")
index2net = pd.read_csv('index2netname.csv', header=None, index_col=0, squeeze=True).to_dict()
n_regions = len(index2net)
# This tupled list only needed for next line
ordered_networks = [(i, index2net[i]) for i in range(n_regions)]
index2rois = {i:np.array([[dfMNI["x"][j], dfMNI["y"][j], dfMNI["z"][j]]
                          for j in np.argwhere(np.array(dfMNI["Net. Name"]) == net).flatten()])
              for i,net in ordered_networks}

m = len(np.argwhere(MASK == 1.0))
print("Size of mask:  " + str(m) + " voxels")

# Iterate over the fourth dimension of time, generating the column wise matrix needed for np.corr_coef
# T - each row is a gray matter voxel time series of BOLD response
T = np.zeros(shape=(m, BOLD_time.shape[3]), dtype=np.float64)
row2MNI = np.zeros(shape=(m, 3), dtype=np.single)
row2voxel = np.zeros(shape=(m, 3), dtype=np.int32)

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

R = np.zeros(shape=(nzero, nzero), dtype=np.single)

start = time.time()
R = getRpthreaded.getRpthreaded(T, R, nzero, BOLD_time.shape[-1])
print("C++ multithreading finished in:  " + str(time.time() - start) + " seconds")

# Assigning structure as volume, -1 defaults to no assignment
voxel_assignVOL = np.zeros(shape=BOLD_time.shape[:3], dtype=np.int32) - 1
print(R[71,71])
print("done!")
