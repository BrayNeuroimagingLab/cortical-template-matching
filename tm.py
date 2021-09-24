import pandas as pd
import numpy as np
import nibabel as nib
import time
import cy_getRpthreaded
import psutil
import ray
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

    :param binarizedTopPerc: the binary volume, shape=(91, 109, 91), dtype=np.bool_ where True
                             indicates the voxel is within the top functionally correlated voxels
                             with the seed voxel
    :return: dices, ndarray of 13 floats, where dices[i] is the ith network degree of overlap
    """
    dices = np.zeros(networkVolumes.shape[0])
    for i in range(networkVolumes.shape[0]):
        binaryNetworkVol = networkVolumes[i]
        AND = np.bitwise_and(binarizedTopPerc, binaryNetworkVol)
        ANDsize = np.count_nonzero(AND)
        dices[i] = ANDsize
    return dices

# Following procedure outlined in Methods, Section 2.3.1, of DOI: 10.1016/j.neuroimage.2021.118164
# Load required params
params = pd.read_csv('parameters.csv', header=None, index_col=0, squeeze=True).to_dict()
BOLD_time = nib.load(params["BOLD_path"]).get_fdata()
MASK = nib.load(params["MASK_path"]).get_fdata()
local_rm_thresh = float(params["local_rm_thresh"])
dice_top_perc = float(params["dice_top_perc"])
AFFINE = np.array([[2.63764, 0, 0, -121],
                   [0, 2.19883, 0.10776, -126],
                   [0, -0.11941, 1.9844, -72],
                   [0, 0, 0, 1]])

# Load ROI data
dfMNI = pd.read_csv("153ProbabilisticROIs_MNI_info.txt", delimiter="\t")
index2net = pd.read_csv('index2netname.csv', header=None, index_col=0, squeeze=True).to_dict()
n_networks = len(index2net)
# This tupled list only needed for next line
ordered_networks = [(i, index2net[i]) for i in range(n_networks)]
index2rois = {i:np.array([[dfMNI["x"][j], dfMNI["y"][j], dfMNI["z"][j]]
                          for j in np.argwhere(np.array(dfMNI["Net. Name"]) == net).flatten()])
              for i,net in ordered_networks}

m = len(np.argwhere(MASK == 1.0))
print("Size of mask before filtering 0 time courses:  " + str(m) + " voxels")

networkVolumes = np.zeros(shape=(n_networks, 91, 109, 91))
template_mask_files = sorted(os.listdir(os.getcwd() + "/BinaryTemplateMasks"))
print("Network binary mask files: ")
print(*template_mask_files, sep="\n")
for i,filename in enumerate((template_mask_files)):
    with open("BinaryTemplateMasks/" + filename, 'rb') as f:
        vol = np.load(f)
    networkVolumes[i] = vol

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
R = cy_getRpthreaded.getRpthreaded_py(T, R, nzero, BOLD_time.shape[-1])
print("C++ multithreading finished in:  " + str(time.time() - start) + " seconds")

# Assigning structure as volume, -1 defaults to no assignment
voxel_assignVOL = np.zeros(shape=BOLD_time.shape[:3], dtype=np.int32) - 1
top_amount = int(nzero * dice_top_perc)

def create_debug_figure(gmv):
    all_correlates = R[:, gmv]
    top5perc = np.argsort(all_correlates)[::-1][:top_amount]
    top5percMNI = np.array(list([row2MNI[gmv]]) + [row2MNI[i] for i in top5perc if
                                                   np.linalg.norm(row2MNI[gmv] - row2MNI[i]) >= local_rm_thresh])
    L = [dice(top5percMNI, index2rois[roi]) for roi in range(n_networks)]
    fig = roi_vis.getFIG(L, BOLD_time)
    fig.add_trace(go.Scatter3d(x=top5percMNI[:, 0],
                               y=top5percMNI[:, 1],
                               z=top5percMNI[:, 2],
                               mode='markers',
                               marker=dict(
                                   size=3,
                                   color=["#5ca832"] + (["#a83248"] * (top5percMNI.shape[0] - 1)),
                                   opacity=0.6
                               ),
                               name="correlates"
                               ))
    fig.write_html("top5perc_correlates-" + str(gmv) + "-.html")

create_debug_figure(33400)
create_debug_figure(12030)
create_debug_figure(15303)

@ray.remote
def assign_worker(lower, upper, R, row2MNI, networkVolumes):
    to_return = np.zeros(shape=(nzero), dtype=np.int32) - 1
    for gmv in range(lower, upper + 1):
        all_correlates = R[:, gmv]
        top5perc = np.argsort(all_correlates)[::-1][:top_amount]
        #top5percMNI = np.array(list([row2MNI[gmv]]) + [row2MNI[i] for i in top5perc if np.linalg.norm(row2MNI[gmv] - row2MNI[i]) >= local_rm_thresh])
        binarizedTopPerc = np.zeros(shape=BOLD_time.shape[:3], dtype=np.bool_)
        for index in top5perc:
            if np.linalg.norm(row2MNI[index] - row2MNI[gmv]) <= local_rm_thresh: continue
            binarizedTopPerc[row2voxel[index]] = True


        dcoefs = dices(binarizedTopPerc, networkVolumes)

        if sum(dcoefs) == 0.0:
            print("All zero dice coefficients were found")
            continue

        assignment = np.argmax(dcoefs)
        to_return[gmv] = assignment
    return to_return

num_cpus = psutil.cpu_count(logical=False)
print("num cpus using: " + str(num_cpus))
ray.init(num_cpus=num_cpus)


R_id = ray.put(np.asarray(R))
row2MNI_id = ray.put(row2MNI)
networkVolumes_id = ray.put(networkVolumes)

load = nzero // num_cpus
print("Load per cpu: " + str(load))

start = time.time()
result_ids = [assign_worker.remote(i * load, (i + 1) * load - 1, R_id, row2MNI_id, networkVolumes_id) if i < num_cpus - 1
              else assign_worker.remote(i * load, nzero - 1, R_id, row2MNI_id, networkVolumes_id) for i in
              range(num_cpus)]
results = np.array(ray.get(result_ids))
print("Finished ray multi remote execution in " + str(time.time() - start))

for gmv in range(nzero):
    assign = np.sort(results[:,gmv])[-1]
    i, j, k = row2voxel[gmv]
    voxel_assignVOL[i, j, k] = assign

print("Unique voxels in volume: " + str(np.unique(voxel_assignVOL)))

# TODO clusters of less than 4 continuous voxels - remove from network map
network_assigned = nib.Nifti1Image(voxel_assignVOL, AFFINE, nib.Nifti1Header())
nib.save(network_assigned, params["output_file"])
print("done!")
