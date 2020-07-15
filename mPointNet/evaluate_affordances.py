'''
    Evaluate classification performance with optional voting.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
HOSTNAME = socket.gethostname()
DATA_DIR= os.path.abspath(os.path.join(ROOT_DIR, 'data/new_data_centered'))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import affordances_h5_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='m_pointnet2', help='Model name. [default: modified pointnet2_cls_ssg (m-pointnet)]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores from multiple rotations [default: 1]')
parser.add_argument('--classes',default='binary',help='binary, miniDataset or Dataset[default: off-centered]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point

GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module

D_TYPE=FLAGS.classes

NUM_CLASSES = 85

SHAPE_NAMES=[]

assert(NUM_POINT<=2048)


# my dataSet splits
# multiclass multilabel
aff_name='All'
name='/MultilabelDataSet_splitTest.h5'
print('Data dir %s'%DATA_DIR)
print('Affordance %s'%aff_name)
print('Classes %d'%NUM_CLASSES)
TEST_FILES=[DATA_DIR+name]
TEST_DATASET=affordances_h5_dataset.AffordancesH5Dataset(TEST_FILES, batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)
name=DATA_DIR+'/all_names.csv'
names=np.genfromtxt(name,dtype='str',delimiter=',')
for intreaction in range(names.shape[0]):
    SHAPE_NAMES.append(names[intreaction])

print("Files: %s"%TEST_FILES)
print (SHAPE_NAMES)

# 'manually set the model '
MODEL_PATH=FLAGS.model_path
DUMP_DIR='dump/'
    

if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
LOG_FOUT.write('CLASSES: '+str(NUM_CLASSES)+'\n')

print(MODEL_PATH)
print(DUMP_DIR)


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate(num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.new_placeholder_inputs(BATCH_SIZE, NUM_POINT,n_classes=NUM_CLASSES)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        # Get model and loss
        pred, sec_pred ,end_points, learned1,l1_indices,l1_sub = MODEL.new_get_model(pointclouds_pl, is_training_pl,n_classes=NUM_CLASSES)
        MODEL.new_get_loss(pred, labels_pl, end_points)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
        'labels_pl': labels_pl,
        'is_training_pl': is_training_pl,
        'pred': pred,
        'sec_pred':sec_pred,
        'learned1':learned1,
        'l1_indices':l1_indices,
        'l1_sub':l1_sub,
        'loss': total_loss}
    # predicted=eval_one_epoch(sess, ops, num_votes)
    predicted, points_sampled_xyz,points_sampled_ids,activation_ids,input_points_ids=eval_one_epoch(sess, ops, num_votes)
    name = DUMP_DIR + "/predicted2"
    np.save(name, predicted)
    name=DUMP_DIR+"/points_sampled"
    np.save(name,points_sampled_xyz)
    print('Pointsets 1: %d %d %d'%(points_sampled_xyz.shape[0],points_sampled_xyz.shape[1],points_sampled_xyz.shape[2]))
    name=DUMP_DIR+"/pointIds"
    np.save(name,points_sampled_ids)
    print('Pointsets 2: %d %d %d'%(points_sampled_ids.shape[0],points_sampled_ids.shape[1],points_sampled_ids.shape[2]))
    name=DUMP_DIR+"/activationIds"
    np.save(name,activation_ids)
    name=DUMP_DIR+"/inputIds"
    np.save(name,input_points_ids)


def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    is_training = False

    # Make sure batch data is of same size
    # cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))
    # cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)


    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE,NUM_CLASSES), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    total_tp=0
    total_fp=0
    total_p=0
    loss_sum = 0
    batch_idx = 0
    total_seen_class = np.zeros(NUM_CLASSES,dtype=np.int32)
    total_correct_class = np.zeros(NUM_CLASSES,dtype=np.int32)
    confusion_m=np.zeros((NUM_CLASSES*2,2),dtype=np.int32)
    instances = TEST_DATASET.size
    # first Pointnet layer activations, 512 -> pointNet++ default input size
    points_sampled_xyz=np.zeros((instances,512,3),dtype=np.float32)
    # ids of points sampled in radius around each point_sampled1
    points_sampled_ids=np.zeros((instances,512,32),dtype=np.int32)
    # #ids of points_sampled_ids that contributed towards feature
    activations=np.zeros((instances,512,128),dtype=np.int32)
    # #ids of inputCloud after shuffle
    input_points=np.zeros((instances,NUM_POINT),dtype=np.int32)
    per_class_tp=np.zeros(NUM_CLASSES,dtype=np.int32)
    per_class_fp=np.zeros(NUM_CLASSES,dtype=np.int32)
    per_class_p=np.zeros(NUM_CLASSES,dtype=np.int32)
    predicted_log=np.zeros((instances,NUM_CLASSES),dtype=np.int32)

    start_i=0
    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES)) # score for classes
        for vote_idx in range(num_votes):
            # Shuffle point order to achieve different farthest samplings
            shuffled_indices = np.arange(NUM_POINT)
            feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                         ops['labels_pl']: cur_batch_label,
                         ops['is_training_pl']: is_training}
            loss_val, pred_val, sec_pred_val,learned1,l1_indices,l1_sub = sess.run([ops['loss'], ops['pred'],
                                                                                    ops['sec_pred'],ops['learned1'],
                                                                                    ops['l1_indices'],ops['l1_sub']],
                                                                                   feed_dict=feed_dict)
            batch_pred_sum += pred_val

        #end_i=(batch_idx+1)*bsize;
        end_i=start_i+bsize
        print('Batch: %03d, batch size: %d st:%d end:%d'%(batch_idx, bsize,start_i,end_i))
        points_sampled_xyz[start_i:end_i,...]=learned1[0:bsize,...]
        points_sampled_ids[start_i:end_i,...]=l1_indices[0:bsize,...]
        activations[start_i:end_i,...]=l1_sub[0:bsize,...]
        input_points[start_i:end_i,...]=shuffled_indices
        #presented_data[start_i:end_i]=
        pred_val=np.round(sec_pred_val)
        for i in range(bsize):
            predicted_pos=np.nonzero(pred_val[i,:])[0]
            actual_pos=np.nonzero(batch_label[i,:])[0]
            tp=np.intersect1d(predicted_pos,actual_pos)
            fp=np.setdiff1d(predicted_pos,actual_pos)
            total_tp+=tp.size
            total_fp+=fp.size
            total_p+=actual_pos.size
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen+=(bsize*NUM_CLASSES)
        loss_sum += loss_val
        batch_idx += 1
        predicted_log[start_i:end_i,...]=pred_val[0:bsize]
        start_i=end_i;
        for i in range(bsize):
            c = cur_batch_label[i]
            l= np.nonzero(cur_batch_label[i])[0]
            t = pred_val[i]
            for k in range(c.size):
                if c[k]==1 and t[k]==1:
                    #true positive
                    #work out index in confusion
                    confusion_m[k*2,0]+=1
                    total_correct_class[k]+=1
                    per_class_tp[k]+=1
                if c[k]==0 and t[k]==1:
                    confusion_m[k*2+1,0]+=1
                if c[k]==0 and t[k]==0:
                    confusion_m[k*2+1,1]+=1
                if c[k]==1 and t[k]==0:
                    confusion_m[k*2,1]+=1
                    per_class_fp[k]+=1
            total_seen_class[l] += 1
            per_class_p[l]+=1
        
    activations=activations[:end_i,...]
    points_sampled_xyz=points_sampled_xyz[:end_i,...]
    points_sampled_ids=points_sampled_ids[:end_i,...]
    input_points=input_points[:end_i,...]
    predicted_log=predicted_log[:end_i,...]
    print(batch_idx)
    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    actual_seen_class=np.array(total_seen_class,dtype=np.float)
    ok_ids=np.nonzero(actual_seen_class)[0]
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class[ok_ids])/np.array(total_seen_class[ok_ids],dtype=np.float))))
    if (total_tp+total_fp)>0:
        log_string('eval precision %f'%(total_tp/float(total_tp+total_fp)))
    else:
        log_string('eval precision 0')
    if total_p>0:
        log_string('eval recall %f'%(total_tp/float(total_p)))
    else:
        log_string('eval recall 0')

    class_accuracies = np.array(total_correct_class[ok_ids])/np.array(total_seen_class[ok_ids],dtype=np.float)
    class_precisions= per_class_tp[ok_ids]/(per_class_tp[ok_ids].astype(float)+per_class_fp[ok_ids].astype(float))
    class_recall=per_class_tp[ok_ids]/per_class_p[ok_ids].astype(float)
    for i, name in enumerate(SHAPE_NAMES):
        #log_string('%10s:\t%0.4f' % (name, class_accuracies[i]))
        if i in set(ok_ids):
            real_i=np.nonzero(ok_ids==i)[0]
            log_string('%10s:\t%0.4f %0.4f' % (name, class_precisions[real_i],class_recall[real_i]))
    # return points_sampled_xyz,points_sampled_ids,activations,input_points
    return predicted_log, points_sampled_xyz,points_sampled_ids,activations,input_points


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=FLAGS.num_votes)
    LOG_FOUT.close()
