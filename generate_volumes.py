import numpy as np
import pandas as pd
import plotly.graph_objects as go
dfMNI = pd.read_csv("153ProbabilisticROIs_MNI_info.txt", delimiter="\t")
index2net = pd.read_csv('index2netname.csv', header=None, index_col=0, squeeze=True).to_dict()
n_regions = len(index2net)
# This tupled list only needed for next line
ordered_networks = [(i, index2net[i]) for i in range(n_regions)]
index2rois = {i:np.array([[dfMNI["x"][j], dfMNI["y"][j], dfMNI["z"][j]]
                          for j in np.argwhere(np.array(dfMNI["Net. Name"]) == net).flatten()])
              for i,net in ordered_networks}

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



# Get the binarized roi volume, time its acquisition
AFFINE = np.array([[2.63764, 0, 0, -121],
                   [0, 2.19883, 0.10776, -126],
                   [0, -0.11941, 1.9844, -72],
                   [0, 0, 0, 1]])

# Uncomment to debug any given network
def saveVolume(index, string):
    rois = index2rois[index]
    ROI_volume = np.zeros(shape=(91, 109, 91), dtype=np.bool_)
    #X = []
    #Y = []
    #Z = []
    for i in range(ROI_volume.shape[0]):
        for j in range(ROI_volume.shape[1]):
            for k in range(ROI_volume.shape[2]):
                mniOfVoxel = scannerspace_from_index(AFFINE, i, j, k)
                for roi in rois:
                    if np.linalg.norm(roi - mniOfVoxel) < 3.5:
                        ROI_volume[i,j,k] = True
                        #X.append(i)
                        #Y.append(j)
                        #Z.append(k)
                        break

    """
    fig = go.Figure(go.Scatter3d(
                 x=X,
                 y=Y,
                 z=Z,
                 mode='markers',
                 marker=dict(
                     size=3,
                     opacity=0.8
                 )))
    fig.show()
    """
    with open("./BinaryTemplateMasks/" + string, 'wb') as f:
        np.save(f, ROI_volume)

for i, net in ordered_networks:
    saveVolume(i, net + "-binary-voxel-mask.npy")
    print("Saved " + str(net))
print("Completed, saved all")
