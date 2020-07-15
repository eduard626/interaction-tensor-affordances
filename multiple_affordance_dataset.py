#  Author: Eduardo Ruiz
#  Date: July 2020

import sys
from io_utils import *
from pointcloud_utils import *
import glob
from progress.bar import Bar
import os
from sklearn.neighbors import BallTree

# size for extracted multil-label pointclouds
points_per_voxel_ = 2048
# the target number of examples per affordance to be used
n_training_examples = 512
# where the single-affordance results are read from
DATA_DIR = 'mPointNet/data/SingleTensors/results/'
# where the individual tensors are read from i.e. descriptors
TENSORS_DIR = 'mPointNet/data/SingleTensors/'
# where the multi-label dataset is to be stored
DATA_MULTI = 'mPointNet/data/new_data_centered/'
dir_dict = {
    "Place": "Placing",
    "Hang": "Hanging",
    "Fill": "Filling",
    "Sit": "Sitting",
}
# some handy IDS to make things more readable
AFF_ID = 3
ORI_ID = 4
SCORE = 5
GOOD_ID = 6
SAMPLE_ID = 7


def find_individual_results(a_scene_id):
    """

    Returns: a list of files with the results of single affordance prediction e.g. Place-bowl, Sit-human

    """
    # get the 'main' file produced by prediction
    base_name = DATA_DIR + '/*' + a_scene_id + '*_3D_*.pcd'
    res_files = glob.glob(base_name)
    return res_files


def read_input_scene_cloud(scene_name):
    # find input scene pointcloud
    scene_files = glob.glob(DATA_DIR + scene_name + '*.pcd')
    # used dense cloud when available
    scene_file = DATA_DIR + scene_name + '_d.pcd'
    if scene_file not in scene_files:
        # try regular file
        scene_file = DATA_DIR + scene_name + '.pcd'
        if scene_file not in scene_files:
            print('Could not find the right input file for {}'.format(scene_name))
            sys.exit()
    pointcloud, _, _=load_pcd_data_binary(scene_file)
    return pointcloud


def read_pairs(all_files):
    labels = []
    query_object_sizes = np.zeros((len(all_files), 1))
    bar = Bar('Reading pairs', max=len(all_files))
    for j in range(len(all_files)):
        keywords = all_files[j].split('/')[-1].split('_')
        affordance_name = keywords[0]
        query_object = keywords[1]
        a_label_set = [dir_dict[affordance_name], affordance_name, query_object]
        labels.append(a_label_set)
        # read object and save size
        query_file = TENSORS_DIR + dir_dict[affordance_name] + '/' + query_object + '.ply'
        query_object_cloud = load_ply_data(query_file)
        # get size using min and max
        query_object_sizes[j, 0] = np.linalg.norm(
            np.max(query_object_cloud, axis=0) - np.min(query_object_cloud, axis=0))
        bar.next()
    bar.finish()
    return np.asarray(labels), query_object_sizes


def compute_common_predictions(scene, res_files, labels, input_cloud, min_z, max_z):
    if not os.path.exists(DATA_MULTI+'common_' + scene + '.h5'):
        # compute data
        useful_counts = np.zeros((len(res_files), 1), dtype=np.int32)
        common = np.zeros((input_cloud.shape[0], len(res_files)), dtype=np.int32)
        negative_common = np.zeros((input_cloud.shape[0], len(res_files)), dtype=np.int32)
        sorted_common = np.zeros((input_cloud.shape[0], len(res_files)), dtype=np.int32) - 1
        common_orientations = np.zeros((input_cloud.shape[0], len(res_files)), dtype=np.int32)
        bar = Bar('Computing common predictions', max=len(res_files))
        for j in range(len(res_files)):
            a_file = res_files[j]
            # get the timestamp that identifies data from this run
            file_id = a_file.split('/')[-1].split('_')[-1].split('.')[0]
            # sampled test-points ids
            tmp_file = DATA_DIR+ file_id + '_samplePointsIds.pcd'
            pc=pypcd.PointCloud.from_path(tmp_file)
            # check sampled points are actually the same for every affordance
            if j > 0:
                assert pc.pc_data['id'].shape[0] == sampled_ids.shape[0], \
                    'Odd? Sample points are not the same with prev affordance'
            sampled_ids = pc.pc_data['id']
            # read 'good' predictions data
            tmp_file = DATA_DIR + file_id + '_goodPointsIds.pcd'
            pc = pypcd.PointCloud.from_path(tmp_file)
            # ids as a subset of sampled ids
            good_ids = pc.pc_data['id'].astype(np.int32)
            # load predicted orientation, affordance, etc for success cases
            tmp_file = DATA_DIR + file_id + '_goodPointsX.pcd'
            data, _, _ = load_pcd_data_binary(tmp_file)
            # load predicted 3d location for success cases, RGB encodes how many affordances predicted per point
            tmp_file = DATA_DIR + file_id + '_goodPoints.pcd'
            points, real_c_data, _ = load_pcd_data_binary(tmp_file)
            red = np.array((real_c_data >> 16) & 0x0000ff, dtype=np.uint8).reshape(-1, 1)
            green = np.array((real_c_data >> 8) & 0x0000ff, dtype=np.uint8).reshape(-1, 1)
            blue = np.array((real_c_data) & 0x0000ff, dtype=np.uint8).reshape(-1, 1)
            real_c_data = np.concatenate((red, green, blue), axis=1)
            perPointDetections = np.sum(real_c_data, axis=1)
            bounds = np.cumsum(perPointDetections)
            start_i = 0
            # array containing all data of 'successful' predictions
            large_data = np.zeros((data.shape[0], 8), dtype=np.float32)
            # store orientation, score and affordance id in cols 3 to 6
            large_data[:, 3:6] = data
            # now we recover 3D coords ot points per affordance per orientation
            for i in range(bounds.size):
                end_i = bounds[i]
                # goodPoint id
                large_data[start_i:end_i, GOOD_ID] = i
                large_data[start_i:end_i, :3] = points[i, :]
                # samplePoint id from 0 to inputCloud size
                large_data[start_i:end_i, SAMPLE_ID] = sampled_ids[good_ids[i]]
                start_i = end_i
            # will delete data to save memory
            del bounds, perPointDetections, data, points, red, green, blue
            # remove data out of target range (height)
            within_height = np.nonzero(large_data[:, 2] >= min_z)[0]
            large_data = large_data[within_height, ...]
            # will use top 25% of results as good predictions#
            # it could be replace by specific threshold score. e.g 0.5
            min_valid = np.percentile(large_data[:, SCORE], 75)
            # use bottom 25% of data as bad or negative examples
            cutoff = np.percentile(large_data[:, SCORE], 25)
            top_ids = np.nonzero(large_data[:, SCORE] >= min_valid)[0]
            bottom_ids = np.nonzero(large_data[:, SCORE] <= cutoff)[0]
            # sort the example predictions according to score
            sorted_top_ids = np.argsort(large_data[top_ids, SCORE])
            sorted_bottom_ids = np.argsort(large_data[bottom_ids, SCORE])
            top_ids = top_ids[sorted_top_ids]
            bottom_ids = bottom_ids[sorted_bottom_ids]
            # reverse top ids to allow order to sample high-scores first
            top_ids = top_ids[::-1]
            # the small datasets to sample from
            top_data = large_data[top_ids, ...]
            bottom_data = large_data[bottom_ids, ...]

            # get the ids of the 3D point in the scene
            point_ids = top_data[:, GOOD_ID]
            # remove duplicates
            _, actual_points_ids = np.unique(point_ids, return_index=True)
            # we only consider this unique predictions and sort it
            top_data = top_data[actual_points_ids, ...]
            sorted_ids = np.argsort(top_data[:, SCORE])
            sorted_ids = sorted_ids[::-1]
            top_data = top_data[sorted_ids, ...]
            # similar thing for negative examples
            point_ids_neg = bottom_data[:, GOOD_ID]
            unique_points_sorted_neg, actual_points_ids_neg = np.unique(point_ids_neg, return_index=True)
            bottom_data = bottom_data[actual_points_ids_neg, :]
            # we care about the point (in the scene) where these predictions were done,
            # 'cause it allows us to extract pointclods for training saliency
            usefull_point_ids = top_data[:, SAMPLE_ID].astype(np.int32)
            # well need orientations to later on map saliency into iTs
            usefull_orientations = top_data[:, ORI_ID]
            # sort the point ids to store them appropriately
            sorted_useful = np.arange(usefull_point_ids.size)
            common[usefull_point_ids, j] = 1
            common_orientations[usefull_point_ids, j] = usefull_orientations
            sorted_common[usefull_point_ids, j] = sorted_useful
            useful_counts[j, 0] = usefull_point_ids.size
            if usefull_point_ids.size < n_training_examples:
                print('\nAffordance {} {} with only {} training examples'.format(labels[j,1], labels[j,2], usefull_point_ids.size))

            # get the negative/non response points
            usefull_point_ids_neg = bottom_data[:, SAMPLE_ID].astype(np.int32)
            negative_common[usefull_point_ids_neg, j] = 1
            bar.next()
        bar.finish()
        # save progress
        print('Saving data')
        save_as_h5(DATA_MULTI+'common_' + scene + '.h5', common)
        save_as_h5(DATA_MULTI+'negative_common_' + scene + '.h5', negative_common)
        save_as_h5(DATA_MULTI+'sorted_common_' + scene + '.h5', sorted_common)
        save_as_h5(DATA_MULTI+'common_orientations_' + scene + '.h5', common_orientations)
        with open(DATA_MULTI+'common_names_' + scene + '.csv', "w") as text_file:
            for i in range(len(res_files)):
                text_file.write("%s,%s,%s\n" % (labels[i,0],labels[i,1],labels[i,2]))
            text_file.write("Non,Non,Non-affordance\n")
    else:
        # read data
        common, _ = load_h5('common_' + scene + '.h5')
        negative_common, _ = load_h5('negative_common_' + scene + '.h5')
        sorted_common, _ = load_h5('sorted_common_' + scene + '.h5')
        common_orientations, _ = load_h5('common_orientations_' + scene + '.h5')
        labels = np.genfromtxt('common_names_' + scene + '.csv', dtype='str', delimiter=',')

    return common, negative_common, sorted_common, common_orientations, labels


def read_results(res_files):
    """
    Reads results from single-affordance predictions
    all_files = All _main_ files being considered for multiple affordance dataset
    """
    # use the first one to get the scene pointcloud
    keywords = res_files[0].split('/')[-1].split('_')
    scene_name = keywords[2]
    # read the input scene cloud
    input_cloud = read_input_scene_cloud(scene_name)
    print('Read input cloud {} with {} points'.format(scene_name,input_cloud.shape))
    # read affodance-object pair names and sizes
    labels, sizes = read_pairs(res_files)
    # will use largest query-object as radius for pointcloud extraction
    max_rad = sizes.max()/2
    print('Read {} affordance-object pairs'.format(labels.shape[0]))
    print('Max query cloud is {}'.format(max_rad))
    print('Done')
    return labels, input_cloud, max_rad


def build_multilabel_data(res_files, input_cloud, max_rad, common, negative_common, sorted_common, common_orientations):
    chk_sum = np.sum(common, axis=1)
    # keep_ids goes from 0 to input_cloud_size
    keep_ids = np.nonzero(chk_sum)[0]
    # check non responses
    chk_sum = np.sum(negative_common, axis=1)
    all_negative = np.nonzero(chk_sum == len(res_files))[0]
    print('Negative examples in data %d' % all_negative.size)
    # common=common[keep_ids,:]
    per_affordance_points = np.sum(common[keep_ids, :], axis=0)
    sorted_ids = np.argsort(per_affordance_points)
    training_examples_per_affordance = np.zeros((len(res_files), 1), dtype=np.int32)
    # save the sampled_id for each point for each affordance so you can recover later the orientation and score, etc
    # create some large arrays to store all data
    data_to_recover = np.zeros((common.shape[0], len(res_files)), dtype=np.uint8)
    data = np.empty((100000, points_per_voxel_, 3), dtype=np.float32)
    data_points = np.empty((100000, 3), dtype=np.float32)
    labels = np.zeros((100000, len(res_files) + 1), dtype=np.uint8)
    # counter for voxels or pointsclouds extracted
    extracted_voxels = 0
    # to check if a test-point has been used previously to create pointclouds
    already_sampled = {}
    # orientations in dataset, useful for backprojections of saliency
    dataSet_orientations = np.zeros((100000, len(res_files)), dtype=np.int32) - 1
    # tree to extract pointcloud from input scene in a radius around test-points
    kdt = BallTree(input_cloud, leaf_size=5, metric='euclidean')
    bar = Bar('Generating dataset', max=len(res_files))
    for i in range(len(res_files)):
        # get points for this affordance
        affordance_id = sorted_ids[i]
        # these are based on input_cloud size
        ids = np.nonzero(common[:, affordance_id])[0]
        # get the order from sorted_matrix
        aff_sorted_ids = sorted_common[ids, affordance_id]
        # this should be increasing, i think no longer makes sense because of what I did with top_ids before
        aff_sorted_ids_ids = np.argsort(aff_sorted_ids)
        # ad point id to dict of already sampled points
        actually_sampled = training_examples_per_affordance[affordance_id, 0]
        j = 0
        # keep creating pointcloud examples until no longer possible
        # and while not achieved the minimum required
        while actually_sampled < n_training_examples and j < ids.size:
            # check point has not been sampled before
            if str(ids[aff_sorted_ids_ids[j]]) not in already_sampled:
                # get the first point
                test_point = input_cloud[ids[aff_sorted_ids_ids[j]], :]
                # extract voxel or pointcloud surrounding the point
                # these are only ids of points in scene cloud
                voxel_ids = getVoxel(test_point, max_rad, kdt)
                # how many points in this pointclouds
                actual_voxel_size = voxel_ids.size
                # Warn if cloud is too sparse
                if actual_voxel_size < points_per_voxel_:
                    print('Bad point? Few points')
                else:
                    # the actual pointcloud
                    voxel = input_cloud[voxel_ids, :]
                    # randomly select a target number of points in this voxel
                    # and 'center' the pointcloud, e.g. origin (0,0,0) is test-point
                    sample = sample_cloud(voxel, points_per_voxel_) - test_point
                    # save data
                    data[extracted_voxels, ...] = sample
                    data_points[extracted_voxels, ...] = test_point
                    # how many affordances here
                    all_responses = np.nonzero(common[ids[aff_sorted_ids_ids[j]], :])
                    # set the label as one-hot vector
                    labels[extracted_voxels, all_responses] = 1
                    # the orientations of these predictions
                    dataSet_orientations[extracted_voxels, all_responses] = common_orientations[
                        ids[aff_sorted_ids_ids[j]], all_responses]
                    training_examples_per_affordance[all_responses, 0] += 1
                    actually_sampled += 1
                    extracted_voxels += 1
                    data_to_recover[ids[aff_sorted_ids_ids[j]], affordance_id] = 1
                    # add to already sample point
                    already_sampled[str(ids[aff_sorted_ids_ids[j]])] = ids[aff_sorted_ids_ids[j]]
            j += 1
        bar.next()
    bar.finish()
    print('Per affordance examples:')
    print(training_examples_per_affordance.T)
    print('Before negatives %d' % extracted_voxels)
    # add 'negative' data
    # if data from input scene has negative examples use those
    # if not, create some random noise
    mean_examples = np.mean(training_examples_per_affordance)
    negatives_to_add = int(mean_examples // 1)
    if all_negative.size > 0:
        if negatives_to_add > all_negative.size:
            negatives_to_add = all_negative.size
        for i in range(negatives_to_add):
            test_point = input_cloud[all_negative[i], ...]
            voxel_ids = getVoxel(test_point, max_rad, kdt)
            actual_voxel_size = voxel_ids.size
            if actual_voxel_size < points_per_voxel_:
                print('Bad point? Few points')
                toGenerate = points_per_voxel_ - actual_voxel_size
                someNoise = genereateNoisyData(np.array([[0, 0, 0]]), max_rad, toGenerate, 1)
                voxel = input_cloud[voxel_ids, :]
                sample = np.concatenate((someNoise, voxel), axis=0)
            else:
                voxel = input_cloud[voxel_ids, :]
                sample = sample_cloud(voxel, points_per_voxel_) - test_point
            data[extracted_voxels, ...] = sample
            data_points[extracted_voxels, ...] = test_point
            labels[extracted_voxels, len(res_files)] = 1
            extracted_voxels += 1
    else:
        for i in range(negatives_to_add):
            someNoise = genereateNoisyData(np.array([[0, 0, 0]]), max_rad, points_per_voxel_, 1)
            sample = someNoise
            data[extracted_voxels, ...] = sample
            data_points[extracted_voxels, ...] = np.array([[0, 0, 0]])
            labels[extracted_voxels, len(res_files)] = 1
            extracted_voxels += 1

    print('After negatives %d' % extracted_voxels)
    data = data[:extracted_voxels, ...]
    data_points = data_points[:extracted_voxels, ...]
    labels = labels[:extracted_voxels, ...]
    print('Saving data')
    orientations = dataSet_orientations[:extracted_voxels, ...]
    name = DATA_MULTI+'MultilabelDataSet_' + scene_name + '_points.h5'
    save_as_h5(name, data_points)
    name = DATA_MULTI+'MultilabelDataSet_' + scene_name + '.h5'
    save_as_h5(name, data, labels, 'float32', 'uint8')
    name = DATA_MULTI+'MultilabelDataSet_' + scene_name + '_Orientations.npy'
    np.save(name, orientations)


def create_dataset_multilabel(scene):
    all_main_files = find_individual_results(scene)
    # sort the files
    all_main_files = sorted(all_main_files)
    # read all files from different affordances(single prediction) in this scene
    labels, input_cloud, target_rad = read_results(all_main_files)
    # Only consider points above min height and below max height
    z_min = input_cloud[:, 2].min() + 0.2
    z_max = input_cloud[:, 2].max() - 0.1
    # compute common test-points and successful predictions across affordances
    common, negative_common, sorted_common, common_orientations, _ = \
        compute_common_predictions(scene, all_main_files, labels, input_cloud, z_min, z_max)
    # use the datapoints in common to build multi label dataset for thi scene
    build_multilabel_data(all_main_files, input_cloud, target_rad, common, negative_common, sorted_common, common_orientations)


if __name__ == '__main__':
    if len(sys.argv)<2:
        print('Need a scene_name e.g. kitchen5')
    else:
        scene_name=sys.argv[1]
        create_dataset_multilabel(scene_name)

