#  Author: Eduardo Ruiz
#  Date:
import sys
import glob
from io_utils import load_pcd_data_binary
from pointcloud_utils import *
import os
import pypcd


DATA = '/home/eduardo/Documents/deep-interaction-tensor/data/'
dtypes_ids = {'names': ('id'), 'formats': ('f4')}
current = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    if len(sys.argv)<2:
        print('Need scene name')
    else:
        scene_name = sys.argv[1]
        if '.pcd' in scene_name or '.ply' in scene_name:
            real_scene=DATA+scene_name
            #remove extension for search
            scene_name=scene_name.split('.')[0]
        else:
            real_scene=DATA+scene_name+'.pcd'
            if not os.path.exists(real_scene):
                real_scene=DATA+scene_name+'_d.pcd'

        base_name = DATA+'*_'+scene_name+'_3D_*.pcd'
        #find results
        all_mains = sorted(glob.glob(base_name))
        # read input scene
        input_cloud, _, _ = load_pcd_data_binary(real_scene)
        print('From {} \nFound {} files'.format(current, len(all_mains)))
        for fname in all_mains:
            file_id = fname.split('/')[-1].split('_')[-1].split('.')[0]
            #sampled points
            sample_file = DATA+file_id+'_samplePoints.pcd'
            sampled_cloud, _, _ = load_pcd_data_binary(sample_file)
            # get ids
            sampled_ids = compute_subset_ids(input_cloud, sampled_cloud)
            assert sampled_cloud.shape[0]==sampled_ids.shape[0], 'Wrong sizes'
            #good points
            good_file = DATA + file_id + '_goodPoints.pcd'
            good_cloud, _, _ = load_pcd_data_binary(good_file)
            # goods ids as subset of sampled
            good_ids = compute_subset_ids(sampled_cloud, good_cloud)
            assert good_cloud.shape[0] == good_ids.shape[0], 'Wrong sizes'
            # print(sampled_cloud.shape, sampled_ids.shape)
            # print(good_cloud.shape,good_ids.shape)
            name = DATA+file_id+'_goodPointsIds.pcd'
            actual_data_array = np.zeros(good_ids.shape[0], dtype=[('id', 'f4')])
            actual_data_array['id'] = good_ids[:, 0]
            new_cloud = pypcd.PointCloud.from_array(actual_data_array)
            new_cloud.save_pcd(name, compression='ascii')
            print('Saved {} data points in {}'.format(good_ids.shape, name))

            name = DATA + file_id + '_samplePointsIds.pcd'
            actual_data_array = np.zeros(sampled_ids.shape[0], dtype=[('id', 'f4')])
            actual_data_array['id'] = sampled_ids[:, 0]
            new_cloud = pypcd.PointCloud.from_array(actual_data_array)
            new_cloud.save_pcd(name, compression='ascii')
            print('Saved {} data points in {}'.format(sampled_ids.shape, name))

