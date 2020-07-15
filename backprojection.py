'''     Author: Eduardo Ruiz
        Date: July 2020
'''

from sklearn.neighbors import KDTree
import os
from progress.bar import Bar
from io_utils import *
from pointcloud_utils import *

DUMP_DIR = 'mPointNet/dump'
DATA_DIR = 'mPointNet/data/new_data_centered'
dir_dict = {
    "Place": "Placing",
    "Hang": "Hanging",
    "Fill": "Filling",
    "Sit": "Sitting",
}

# cell size for uniform grid in meters
cell_size = 0.01
# a descriptor id, currently in the 900's is accepted
descriptor_id = 994
# Read affordances/tensors with the following names/pairs
names = np.genfromtxt(DATA_DIR+'/all_names.csv', dtype='str')
# Number of affordances, without non-affordance class
n_affordances = names.size-1
# hardcoded size of largest affordance in our study, e.g. Sit-human
max_rad = 0.806884  # in meters


def recoverSaliency():
    global n_affordances, names, max_rad

    name = DATA_DIR + '/MultilabelDataSet_splitTest.h5'
    input_clouds, input_labels = load_h5(name)
    name = DUMP_DIR + '/pointIds.npy'
    pointSampledIds = np.load(name)
    name = DUMP_DIR + '/activationIds.npy'
    activationIds = np.load(name)
    name = DUMP_DIR + '/points_sampled.npy'
    pointSampled = np.load(name)
    input_ids_name = DUMP_DIR + '/inputIds.npy'
    inputIds = np.load(input_ids_name)
    # by default evaluate_affordances.py takes the first NUM_POINT points
    # which can be known by inputIds dim 1
    input_clouds = input_clouds[:,:inputIds.shape[1],...]
    # input_labels = input_labels[:inputIds.shape[1], ...]

    # all activations have at most 128 points with ids within [0-32)
    all_activations = np.zeros((input_clouds.shape[0], pointSampled.shape[1], 128), dtype=np.int32)
    print('Input clouds')
    print(input_clouds.shape)
    print('Input Ids')
    print(inputIds.shape)
    print('Points sampled')
    print(pointSampled.shape)
    print('Sampled ids')
    print(pointSampledIds.shape)
    print('Activations')
    print(activationIds.shape)
    # For every point sampled in the first layer, recover the ids wrt to input pointcloud
    bar = Bar('Recovering data ',max=input_clouds.shape[0])
    for j in range(input_clouds.shape[0]):
        # id of points sampled for this input pointcloud
        sampled = pointSampledIds[j,...]
        # id of points that accounted for the max-pooled features of this input pointcloud
        activated=activationIds[j,...]
        # for every activated (max-pooled) id
        for i in range(activated.shape[0]):
            # vector with 128 ids
            point_ids_per_sample=activated[i,:]
            # get the 'global' ids of these local-region ids
            all_activations[j,i,...]=sampled[i,point_ids_per_sample]
        bar.next()
    bar.finish()
    print('All activations {}'.format(all_activations.shape))
    # verify that largest id is <= input cloud size
    assert all_activations.max() <= inputIds.max(), "Id larger than input"
    # Uncomment the following lines if need to release RAM
    # del pointSampled,pointSampledIds,activationIds
    # np.save(new_name,all_activations)
    # Recover orientations from dataset
    name = DATA_DIR + '/MultilabelDataSet_splitTest.npy'
    test_ids = np.load(name)
    # all affordances classes - nonAffordance -> 84 classes
    orientations = np.zeros((input_labels.shape[0], n_affordances))
    files = ['/MultilabelDataSet_kitchen5_Orientations.npy', '/MultilabelDataSet_living-room6_Orientations.npy']
    files = [DATA_DIR+f for f in files]
    lim_low=0
    bar = Bar('Recovering orientations ', max=len(files))
    for i in range(len(files)):
        fname = files[i]
        # get the orientations in this file
        some_testing_orientations=np.load(fname)
        # get the ids of the testing set that were extracted from this file
        # ids of testing fro mthis file should be between these limits
        lim_high = lim_low+some_testing_orientations.shape[0]
        orientations_above = (test_ids >= lim_low)
        orientations_below = (test_ids < lim_high)
        orientations_from_file = np.logical_and(orientations_above, orientations_below)
        orientations_here_ids = np.nonzero(orientations_from_file)[0]
        # store this orientations into the larger array
        orientations[orientations_here_ids, ...] = some_testing_orientations[test_ids[orientations_here_ids]-lim_low, ...]
        # update lims
        lim_low = lim_high
        bar.next()
    bar.finish()
    return all_activations, orientations, inputIds, input_clouds, input_labels


def downSampleAllTensors():
    # ids of populated cells from grid, i.e. downsampled version of the tensors
    downsampled_tensors_file = DATA_DIR+'/downsampled'+str(n_affordances)+'tensors.npy'
    if not os.path.exists(downsampled_tensors_file):
        # Now we fit a grid (i.e. downsample) to individual tensors before projecting activations
        # build a grid and tree
        print('Bulding grid')
        x_ = np.arange(-max_rad, max_rad, cell_size)
        y_ = np.arange(-max_rad, max_rad, cell_size)
        z_ = np.arange(-max_rad, max_rad, cell_size)
        x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        z = z.reshape(1, -1)
        grid = np.concatenate((x, y, z), axis=0).T
        print('Done')

        print('Building tree ')
        kdt = KDTree(grid, metric='euclidean', )
        print('Done')
        cell_ids = np.zeros((grid.shape[0], n_affordances), dtype=np.int32)
        bar = Bar('Downsampling Tensors ', max=n_affordances)
        object_sizes = np.zeros((n_affordances, 1), dtype=np.float32)
        for i in range(n_affordances):
            tokens = names[i].split('-')
            aff = tokens[0]
            obj = '-'.join(tokens[1:])
            dirr = dir_dict[aff]
            tensor_file = DATA_DIR+'/SingleTensors/' + dirr + '/' + aff + '_' + obj + '_field_clean.pcd'
            # read tensor cloud
            cloud, _, _ = load_pcd_data_binary(tensor_file)
            data_file = DATA_DIR+'/SingleTensors/' + dirr + '/ibs_full_' + aff + '_' + obj + '.txt'
            # read data file -> scene point to translate everything
            test_point = read_training_sample_point(data_file)
            # read object size for later clipping
            object_cloud_file = DATA_DIR+'/SingleTensors/' + dirr + '/' + obj + '.ply'
            o_points = load_ply_data(object_cloud_file)
            maxP = np.max(o_points, axis=0).reshape(1, -1)
            minP = np.min(o_points, axis=0).reshape(1, -1)
            a_size = np.linalg.norm(maxP - minP, axis=1)
            object_sizes[i, 0] = a_size
            # translate cloud back to origin
            cloud = cloud - test_point
            # clip pointcloud inside sphere with a_size radi
            distances = np.linalg.norm(cloud, axis=1)
            inside_sphere = np.nonzero(distances <= (a_size / 2))[0]
            cloud = cloud[inside_sphere, :]
            # fit the grid to the tensor and get cells
            _, ind = kdt.query(cloud, k=1)
            real_activations = np.unique(ind[:, 0])
            cell_ids[real_activations, i] += 1
            bar.next()
        bar.finish()
        print(cell_ids.shape)
        np.save(downsampled_tensors_file, cell_ids)
    else:
        print('Reading downsampled Tensors')
        cell_ids = np.load(downsampled_tensors_file)

    return cell_ids



def projectSaliency(cell_ids, all_activations, orientations, inputIds, input_clouds, input_labels):
    saliency_projections_file = DATA_DIR+'/back_projected_saliency.npy'
    if not os.path.exists(saliency_projections_file):
        # build a grid and tree
        print('Bulding grid')
        x_ = np.arange(-max_rad, max_rad, cell_size)
        y_ = np.arange(-max_rad, max_rad, cell_size)
        z_ = np.arange(-max_rad, max_rad, cell_size)
        x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        z = z.reshape(1, -1)
        grid = np.concatenate((x, y, z), axis=0).T
        print('Done')
        back_projection_votes = np.zeros(cell_ids.shape, dtype=np.int16)
        bar = Bar('Saliency backprojection', max=all_activations.shape[0])
        for i in range(all_activations.shape[0]):
            # rotations in this batch of salient points
            rotations = orientations[i, ...]
            # id of this batch of salient points
            thisActivations = all_activations[i, ...]
            # these could be repeated due to overlapping sampling regions of pointNet++
            # thus, ignore duplicates
            real_ids = np.unique(thisActivations)
            # ids wrt to input cloud
            real_ids = inputIds[i, real_ids]
            # 3D points of these ids
            # print(input_clouds.shape[1],real_ids.min(),real_ids.max())
            activations_3d = input_clouds[i, real_ids, :]
            # affordances in this batch of scene saliency
            affordances_actually_here = np.nonzero(input_labels[i, :n_affordances])[0]
            for j in range(affordances_actually_here.size):
                # an affordance in this pointcloud
                affordance_id = affordances_actually_here[j]
                # rotate the activations back into common frame
                # the angle in radians
                anAngle = -rotations[affordance_id] * (2 * np.pi) / 8
                # the actual pointcloud of 3D saliency
                saliency_cloud = rotate_point_cloud_by_angle(activations_3d, anAngle)
                # get the downsampled tensor of this affordance by means of their 1cm-cell ids
                downsampled_ids = np.nonzero(cell_ids[:, affordance_id])[0]
                # build a search tree only with the cells representing the downsampled tensor
                downsampled_tensor = grid[downsampled_ids, :]
                kdt = KDTree(downsampled_tensor, metric='euclidean')
                # get the closest cell-centroid for every 3D salient point, i.e. backprojection
                _, ind = kdt.query(saliency_cloud, k=1)
                # ids could be repeated, ignore repeats
                ind = np.unique(ind[:, 0])
                # increase the counter of these 'activated' cells, i.e. voting
                back_projection_votes[downsampled_ids[ind], affordance_id] += 1
            bar.next()
        bar.finish()
        np.save(saliency_projections_file, back_projection_votes)
    else:
        print('Reading all projected saliency')
        back_projection_votes = np.load(saliency_projections_file)
        print('Done {}'.format(back_projection_votes.shape))

    tensor_data_file = DATA_DIR+'/all_tensors_traninig_test_points,h5'
    if not os.path.exists(tensor_data_file):
        bar = Bar('Will load tensor data', max=n_affordances)
        # read all tensors useful data
        all_tensor_data = []
        all_training_test_ponints = np.zeros((n_affordances,3))
        for j in range(n_affordances):
            an_affordance_id = j
            # build id the name of this affordance using the id
            tokens = names[an_affordance_id].split('-')
            aff = tokens[0]
            obj = '-'.join(tokens[1:])
            dirr = dir_dict[aff]
            # read the data for this tensor
            tensor_file = DATA_DIR + '/SingleTensors/' + dirr + '/' + aff + '_' + obj + '_field_clean.pcd'
            # read tensor keypoints, 3d points and provenance vectors
            tensor_3d_points, _, provenance_vectors = load_pcd_data_binary(tensor_file)
            data_file = DATA_DIR + '/SingleTensors/' + dirr + '/ibs_full_' + aff + '_' + obj + '.txt'
            # read data file -> scene point to translate everything
            test_point = read_training_sample_point(data_file)
            all_tensor_data.append(np.concatenate((tensor_3d_points,provenance_vectors), axis=1).tolist())
            # all_tensor_data[an_affordance_id,:3] = tensor_3d_points
            # all_tensor_data[an_affordance_id, 3:6] = provenance_vectors
            all_training_test_ponints[an_affordance_id,:] = test_point
            bar.next()
        bar.finish()
        save_as_h5(tensor_data_file,all_training_test_ponints)
    else:
        all_training_test_ponints,_ = load_h5(tensor_data_file)

    mutiple_affordance_data_file=DATA_DIR+'/agglo_all_data_clipped.h5'
    if not os.path.exists(mutiple_affordance_data_file):
        # get the instances per affordance in the dataset
        instances_in_dataset=np.count_nonzero(input_labels[:, :n_affordances], axis=0)
        # compute the cells (downsampled tensors) that received at least 50% of saliency projections
        common_cells = np.zeros(back_projection_votes.shape, dtype=np.int8)
        for i in range(n_affordances):
            # average response per cell in this affordance
            this_responses = back_projection_votes[:, i] / float(instances_in_dataset[i])
            # get only those with at least half 50%
            pop = np.nonzero(this_responses >= 0.5)[0]
            # mark these ids with 1 for later recovery
            common_cells[pop, i] = 1
        # del responses, all_activations
        # del back_projection_votes, all_activations

        # Recover only those cells marked with 1, i.e. with at least 50% of projections
        tmp = np.count_nonzero(common_cells, axis=1)
        most_voted_cells_ids = np.nonzero(tmp)[0]
        # We now only consider the uniform-size grid cell comprised by these 'activated' cells
        smaller_grid = grid[most_voted_cells_ids, :]
        common_cells = common_cells[most_voted_cells_ids, ...]
        # An aux array to save centroids
        agglo_points = np.zeros(smaller_grid.shape)
        # the size of the data
        real_size = np.sum(np.sum(common_cells, axis=1))
        # affordance keypoints -> 3D point and vector
        all_data = np.empty((real_size, 6))
        # some useful extra data: affordance id, orientation id, min/max magnitude of provenance vectors
        all_data_extra = np.empty((real_size, 5))
        # number of affordances per point in the multiple-affordance data
        agglo_data = np.zeros((agglo_points.shape[0], 1), dtype=np.int32)
        start_i = 0
        bar = Bar('Updating mutliple-affordance data', max=common_cells.shape[0])
        # read again checking common points in every cell
        for i in range(common_cells.shape[0]):
            cell_activations = np.nonzero(common_cells[i, :])[0]
            # for every affordance contained by this cell, find 1-NN in every tensor and update centroid
            # cell centroid before update, i.e. from uniform-cell grid
            cell_centre = smaller_grid[i, :].reshape(1, -1)
            # aux array for tensor data in this cell
            cell_data = np.zeros((cell_activations.size, 6), dtype=np.float32)
            # aux array for extra data in this cell, e.g. affordance_id, orientation,etc
            cell_data_extra = np.zeros((cell_activations.size, 5), dtype=np.float32)
            # number of affordances in this cell
            agglo_data[i, 0] = cell_activations.size
            end_i = start_i + cell_activations.size
            for j in range(cell_activations.size):
                an_affordance_id = cell_activations[j]
                data = np.array(all_tensor_data[an_affordance_id])
                # print(data.shape)
                tensor_3d_points = data[:,:3]
                provenance_vectors = data[:,3:6]
                # compute magnitude of provenance vectors
                provenance_vectors_mags = np.linalg.norm(provenance_vectors, axis=1)
                test_point = all_training_test_ponints[an_affordance_id, :]
                # translate tensor back to common origin back substracting test-point
                tensor_3d_points = tensor_3d_points - test_point
                # get the 1-NN keypoint for every cell
                kdt = KDTree(tensor_3d_points, metric='euclidean')
                _, ind = kdt.query(cell_centre, k=1)
                keypoint_id = ind[0, 0]
                # store 3d point and vector in bigger array
                cell_data[j, :3] = tensor_3d_points[keypoint_id, :]
                cell_data[j, 3:] = provenance_vectors[keypoint_id, :]
                # id from tensor
                cell_data_extra[j, 0] = keypoint_id
                # id of orientation
                cell_data_extra[j, 1] = 0
                # id of affordance
                cell_data_extra[j, 2] = an_affordance_id
                # max vector in tensor
                cell_data_extra[j, 3] = np.max(provenance_vectors_mags)
                # min vector in tensor
                cell_data_extra[j, 4] = np.min(provenance_vectors_mags)
            # update cell centroid
            agglo_points[i, :] = np.mean(cell_data[:, :3], axis=0)
            all_data[start_i:end_i, ...] = cell_data
            all_data_extra[start_i:end_i, ...] = cell_data_extra
            start_i = end_i
            bar.next()
        bar.finish()
        # save 'progress'
        save_as_h5(mutiple_affordance_data_file, all_data, 'float32')
        save_as_h5(DATA_DIR+'/agglo_all_data_extra_clipped.h5', all_data_extra, 'float32')
        save_as_h5(DATA_DIR+'/agglo_points_clipped.h5', agglo_points, 'float32')
        save_as_h5(DATA_DIR+'/agglo_data_clipped.h5', agglo_data, 'int32')
    else:
        print('Reading agglo data')
        all_data,_ = load_h5(mutiple_affordance_data_file)
        all_data_extra,_ = load_h5(DATA_DIR+'/agglo_all_data_extra_clipped.h5')
        agglo_points,_ = load_h5(DATA_DIR+'/agglo_points_clipped.h5')
        agglo_data,_ = load_h5(DATA_DIR+'/agglo_data_clipped.h5')

    return all_data, agglo_points, all_data_extra, agglo_data


def buildDescriptor(all_data, agglo_points, all_data_extra, agglo_data):
    global n_affordances, descriptor_id, names
    # We now have a multiple affordance descriptor for 1-orientations
    # For testing/inference, it is more efficient to try multiple orientations at once
    # Thus, rotate the data 8 times, as in the paper.
    n_orientations = 8
    bigger_data_points = np.empty((all_data.shape[0] * n_orientations, 3))
    bigger_agglo_points = np.empty((agglo_points.shape[0] * n_orientations, 3))
    start_i1 = 0
    start_i2 = 0
    orientation_id = np.zeros((all_data_extra.shape[0] * n_orientations, 1))
    for i in range(n_orientations):
        end_i1 = start_i1 + all_data.shape[0]
        angle = i * (2 * np.pi / n_orientations)
        bigger_data_points[start_i1:end_i1, ...] = rotate_point_cloud_by_angle(all_data[:, :3], angle)
        orientation_id[start_i1:end_i1, 0] = i
        end_i2 = start_i2 + agglo_points.shape[0]
        bigger_agglo_points[start_i2:end_i2, ...] = rotate_point_cloud_by_angle(agglo_points, angle)
        start_i2 = end_i2
        start_i1 = end_i1

    # Build some pointclouds needed by the prediction (c++)
    # centroids for NN search
    name = DATA_DIR + '/New' + str(descriptor_id) + '_Approx_descriptor_'+str(n_orientations)+'.pcd'
    actual_data_array = np.zeros(bigger_agglo_points.shape[0], dtype={'names': ('x', 'y', 'z'),
                                                                      'formats': ('f4', 'f4', 'f4')})
    actual_data_array['x'] = bigger_agglo_points[:, 0]
    actual_data_array['y'] = bigger_agglo_points[:, 1]
    actual_data_array['z'] = bigger_agglo_points[:, 2]
    new_cloud = pypcd.PointCloud.from_array(actual_data_array)
    new_cloud.save_pcd(name, compression='ascii')
    print('Saved {}'.format(name))
    # keypoints per cell
    name = DATA_DIR + '/New' + str(descriptor_id) + '_Approx_descriptor_'+str(n_orientations)+'_members.pcd'
    new_agglo_data = np.expand_dims(np.tile(agglo_data[:, 0], n_orientations), axis=1)
    cum_sum = np.cumsum(new_agglo_data, axis=0) - new_agglo_data
    actual_data_array = np.zeros(new_agglo_data.shape[0], dtype={'names': ('x', 'y', 'z'),
                                                                 'formats': ('f4', 'f4', 'f4')})
    actual_data_array['x'] = new_agglo_data[:, 0]
    actual_data_array['y'] = cum_sum[:, 0]
    actual_data_array['z'] = 0
    new_cloud = pypcd.PointCloud.from_array(actual_data_array)
    new_cloud.save_pcd(name, compression='ascii')
    print('Saved {}'.format(name))

    # extra info -> affordance_id,orientation_id,prpvenance_id
    name = DATA_DIR + '/New' + str(descriptor_id) + '_Approx_descriptor_'+str(n_orientations)+'_extra.pcd'
    actual_data_array = np.zeros(bigger_data_points.shape[0], dtype={'names': ('x', 'y', 'z'),
                                                                     'formats': ('f4', 'f4', 'f4')})
    actual_data_array['x'] = np.tile(all_data_extra[:, 2] + 1, n_orientations)
    actual_data_array['y'] = orientation_id[:, 0]
    actual_data_array['z'] = 0
    new_cloud = pypcd.PointCloud.from_array(actual_data_array)
    new_cloud.save_pcd(name, compression='ascii')
    print('Saved {}'.format(name))
    # raw points -> all 3d locations represented by the multiple affordance descriptor
    name = DATA_DIR + '/New' + str(descriptor_id) + '_Approx_descriptor_'+str(n_orientations)+'_points.pcd'
    actual_data_array = np.zeros(bigger_data_points.shape[0], dtype={'names': ('x', 'y', 'z'),
                                                                     'formats': ('f4', 'f4', 'f4')})
    actual_data_array['x'] = bigger_data_points[:, 0]
    actual_data_array['y'] = bigger_data_points[:, 1]
    actual_data_array['z'] = bigger_data_points[:, 2]
    new_cloud = pypcd.PointCloud.from_array(actual_data_array)
    new_cloud.save_pcd(name, compression='ascii')
    print('Saved {}'.format(name))
    # vdata -> provenance vectors mags, weights
    # first need to remap/normalize weights
    # names = np.expand_dims(np.genfromtxt('common_namesreal-kitchen1.csv', dtype='str'), axis=1)
    # all_names = np.empty((names.shape[0], 3), dtype='object')
    # print('Total Affordances %d' % (n_affordances))
    point_counts_data = np.empty((n_affordances, n_orientations), dtype=np.float32)
    vdata = np.empty((all_data.shape[0], 2))
    for i in range(n_affordances):
        # get the points in this affordance
        ids = np.nonzero(all_data_extra[:, 2] == i)[0]
        # get the max and min values
        maxV = np.max(all_data_extra[ids, 3])
        minV = np.min(all_data_extra[ids, 4])
        vectors = all_data[ids, 3:]
        vectors_norm = np.linalg.norm(vectors, axis=1)
        weights = (vectors_norm - minV) * ((1 - 0) / (maxV - minV)) + 0
        vdata[ids, 0] = vectors_norm
        vdata[ids, 1] = 1 - weights
        # get also the per-affordance points to build the point_counts file
        point_counts_data[i, :] = ids.size

    name = DATA_DIR + '/New' + str(descriptor_id) + '_Approx_descriptor_'+str(n_orientations)+'_vdata.pcd'
    actual_data_array = np.zeros(bigger_data_points.shape[0], dtype={'names': ('x', 'y', 'z'),
                                                                     'formats': ('f4', 'f4', 'f4')})
    actual_data_array['x'] = np.tile(vdata[:, 0], n_orientations)
    actual_data_array['y'] = np.tile(vdata[:, 1], n_orientations)
    actual_data_array['z'] = 0
    new_cloud = pypcd.PointCloud.from_array(actual_data_array)
    new_cloud.save_pcd(name, compression='ascii')
    print(name)
    # raw vectors -> provenance vectors
    name = DATA_DIR + '/New' + str(descriptor_id) + '_Approx_descriptor_'+str(n_orientations)+'_vectors.pcd'
    actual_data_array = np.zeros(bigger_data_points.shape[0], dtype={'names': ('x', 'y', 'z'),
                                                                     'formats': ('f4', 'f4', 'f4')})
    actual_data_array['x'] = np.tile(all_data[:, 3], n_orientations)
    actual_data_array['y'] = np.tile(all_data[:, 4], n_orientations)
    actual_data_array['z'] = np.tile(all_data[:, 5], n_orientations)
    new_cloud = pypcd.PointCloud.from_array(actual_data_array)
    new_cloud.save_pcd(name, compression='ascii')
    print('Saved {}'.format(name))
    # per affordance keypoint count, odd format but needed for prediction
    name = DATA_DIR + '/point_count' + str(descriptor_id) + '.dat'
    f = open(name, 'w+b')
    b = np.array(point_counts_data.shape, dtype=np.uint32).reshape(1, -1)
    b = np.fliplr(b)
    binary_format = bytearray(b)
    f.write(binary_format)
    binary_format = bytearray(point_counts_data.T)
    f.write(binary_format)
    f.close()
    print('Saved {}'.format(name))
    name = DATA_DIR + '/tmp' + str(descriptor_id) + '.csv'
    with open(name, "w") as text_file:
        text_file.write("Directory,Affordance,Object\n")
        for i in range(n_affordances):
            tokens = names[i].split('-')
            text_file.write("%s,%s,%s\n" % (dir_dict[tokens[0]], tokens[0], tokens[1]))
    print('Saved {}'.format(name))


if __name__ == '__main__':
    all_activations, orientations, inputIds, input_clouds, input_labels = recoverSaliency()
    cell_ids = downSampleAllTensors()
    all_data, agglo_points, all_data_extra, agglo_data = projectSaliency(cell_ids, all_activations,
                                                                         orientations, inputIds,
                                                                         input_clouds, input_labels)
    buildDescriptor(all_data, agglo_points, all_data_extra, agglo_data)

##SplitTest1 -> kitchen5+living-room6
##SplitTest2 -> kitchen5+real-kitchen1
##SplitTest3 -> kitchen5+real-kitchen+living-room6
##SplitTest4 -> real-kitchen1 + living-room6
##splitTest5 -> real-kitchen1 + real-kitchen2


