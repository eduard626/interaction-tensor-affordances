#  Author: Eduardo Ruiz
#  Date: July 2020
import numpy as np
from sklearn.neighbors import KDTree

# from Charles Qi
def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
    Input:
    BxNx3 array, original batch of point clouds
    Return:
    BxNx3 array, rotated batch of point clouds"""
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval,0],
                                   [-sinval, cosval,0],
                                   [0, 0, 1]])
    shape_pc = batch_data
    rotated_data = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def compute_subset_ids(input_cloud, sampled_cloud):
    """ Finds 1-NN in input_clout for every point in sampled_cloud

    Returns: ids of sampled_points wrt to input_cloud
    """
    kdt = KDTree(input_cloud, metric='euclidean')
    _, ind = kdt.query(sampled_cloud, k=1)
    return ind

def getVoxel(seedPoint,rad,tree):
    ind = tree.query_radius(seedPoint.reshape(1,-1),r=rad)
    point_ids=np.expand_dims(ind,axis=0)[0,0].reshape(1,-1)
    return point_ids[0, :]


def sample_cloud(data,n_samples):
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    return data[idx[0:n_samples], :]


def genereateNoisyData(anchor,rad,d_points,x_samples):
    low=anchor[0,0]-rad
    high=anchor[0,0]+rad
    tmp1=(high - low) * np.random.random_sample((x_samples,d_points, 1)) + (low)
    low=anchor[0,1]-rad
    high=anchor[0,1]+rad
    tmp2=(high - low) * np.random.random_sample((x_samples,d_points, 1)) + (low)
    low=anchor[0,2]-rad
    high=anchor[0,2]+rad
    tmp3=(high - low) * np.random.random_sample((x_samples,d_points, 1)) + (low)
    data=np.concatenate((tmp1,tmp2,tmp3),axis=2)
    return data



