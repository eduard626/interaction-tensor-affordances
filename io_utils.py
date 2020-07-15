#  Author: Eduardo Ruiz
#  Date: July 2020

from plyfile import PlyData
import pypcd
import h5py
import numpy as np


def load_h5(h5_filename):
    f = h5py.File(h5_filename,'r')
    data = f['data'][:]
    if 'label' in f.keys():
        label = f['label'][:]
        return data, label
    else:
        return data, None


def save_as_h5(h5_filename, data, label=None, data_dtype='float32', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    if label is not None:
        h5_fout.create_dataset(
                'label', data=label,
                compression='gzip', compression_opts=1,
                dtype=label_dtype)
    h5_fout.close()


def load_pcd_data_binary(filename):
    pc=pypcd.PointCloud.from_path(filename)
    xyz = np.empty((pc.points, 3), dtype=np.float)
    rgb=np.empty((pc.points, 1), dtype=np.int)
    normals=np.empty((pc.points, 3), dtype=np.float)
    xyz[:, 0] = pc.pc_data['x']
    xyz[:, 1] = pc.pc_data['y']
    xyz[:, 2] = pc.pc_data['z']
    try:
        rgb = pc.pc_data['rgb']
    except Exception as e:
        error_msg=e
    try:
        normals[:,0]=pc.pc_data['normal_x']
        normals[:,1]=pc.pc_data['normal_y']
        normals[:,2]=pc.pc_data['normal_z']
    except Exception as e:
        error_msg=e

    return xyz,rgb,normals


def read_training_sample_point(data_file):
    with open(data_file) as f:
            content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    tmp=content[8].split(":")[1]
    datapoint=tmp.split(',')
    test_point=np.expand_dims(np.asarray([float(x) for x in datapoint]),axis=0)
    return test_point


def load_ply_data(filename):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array


def save_point_cloud_data(data, fname, dtypes=None):
    if dtypes is None:
        dtypes = {'names': ('x', 'y', 'z'), 'formats': ('f4', 'f4', 'f4')}
    if (len(dtypes['names']) != len(dtypes['formats']))\
            or (len(dtypes['names']) != data.shape[1])\
            or (len(dtypes['formats']) != data.shape[1]):
        RuntimeError('Inconsistent field names, formats and data')
    actual_data_array = np.zeros(data.shape[0], dtype=dtypes)
    for j in range(data.shape[1]):
        actual_data_array[dtypes['names'][j]] = data[:, j]
    new_cloud = pypcd.PointCloud.from_array(actual_data_array)
    new_cloud.save_pcd(fname, compression='ascii')
    print('Saved {} data points in {}'.format(data.shape, fname))



