'''
    Single-GPU training.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
import affordances_h5_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
DATA_DIR= os.path.abspath(os.path.join(ROOT_DIR, 'data/new_data_centered'))
print(DATA_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='m_pointnet2', help='Model name. [default: modified pointnet2_cls_ssg (m-pointnet)]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=251, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--data',default='centered',help='centered or off-centered[default: centered]')
parser.add_argument('--classes',default='binary',help='binary, miniDataset1 or miniDataset2[default: off-centered]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
DATA=FLAGS.data
D_TYPE=FLAGS.classes

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_affordances.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 85

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# miniDataset3 is multiple affordances with multiple labels
# multiclass multilabel
name='/MultilabelDataSet_splitTrain.h5'
TRAIN_FILES=[DATA_DIR+name]
TRAIN_DATASET = affordances_h5_dataset.AffordancesH5Dataset(TRAIN_FILES, batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=True)
name='/MultilabelDataSet_splitTest.h5'
print('Data dir %s'%DATA_DIR)
print('Classes %d'%NUM_CLASSES)
TEST_FILES=[DATA_DIR+name]
TEST_DATASET=affordances_h5_dataset.AffordancesH5Dataset(TEST_FILES, batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)

print("DATA %s"%DATA)
print("Train Files: %s"%TRAIN_FILES)
print("Test Files: %s"%TEST_FILES)


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.new_placeholder_inputs(BATCH_SIZE, NUM_POINT, n_classes=NUM_CLASSES)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, sec_pred ,end_points,_, _, _= MODEL.new_get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay,n_classes=NUM_CLASSES,wdecay=0.01)
            MODEL.new_get_loss(pred, labels_pl, end_points)
            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)

            correct = tf.equal(tf.round(sec_pred),labels_pl)
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_CLASSES)
            tf.summary.scalar('accuracy', accuracy)

            print "--- Get training operator"
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
            'labels_pl': labels_pl,
            'is_training_pl': is_training_pl,
            'pred': pred,
            'sec_pred':sec_pred,
            'loss': total_loss,
            'train_op': train_op,
            'merged': merged,
            'step': batch,
            'end_points': end_points,
           }

        early_stop=False
        print('Train ',TRAIN_DATASET.size)
        print('Test ', TEST_DATASET.size)
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)
            # Save the variables to disk.
            if epoch % 10 == 0 or early_stop:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

            if early_stop:
                break


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TRAIN_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE, NUM_CLASSES), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    total_tp=0
    total_fp=0
    total_p=0
    loss_sum = 0
    batch_idx = 0
    while TRAIN_DATASET.has_next_batch():
        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=False)
        # batch_data = provider.random_point_dropout(batch_data)
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label
        print("BATCH {} [{}/{}]".format(batch_idx, (batch_idx+1)*bsize, TRAIN_DATASET.size))
        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training,}

        summary, step, _, loss_val, pred_val,sec_pred_val= sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred'],ops['sec_pred']], feed_dict=feed_dict)

        train_writer.add_summary(summary, step)
        pred_val=np.round(sec_pred_val)

        predicted_pos=np.nonzero(pred_val[0:bsize]==0)[0]
        actual_pos=np.nonzero(batch_label[0:bsize]==0)[0]
        tp=np.intersect1d(predicted_pos,actual_pos)
        fp=np.setdiff1d(predicted_pos,actual_pos)
        total_tp+=tp.size
        total_fp+=fp.size
        total_p+=actual_pos.size
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += (bsize*NUM_CLASSES)
        loss_sum += loss_val
        batch_idx += 1
    log_string('mean loss: %f' % (loss_sum / float(batch_idx)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))
    if (total_tp+total_fp)>0:
        log_string('precision %f'%(total_tp/float(total_tp+total_fp)))
    else:
        log_string('precision 0')
    if total_p>0:
        log_string('recall %f'%(total_tp/float(total_p)))
    else:
        log_string('recall 0')
    TRAIN_DATASET.reset()

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE,NUM_CLASSES), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    total_tp=0
    total_fp=0
    total_p=0
    total_seen_class = np.zeros(NUM_CLASSES,dtype=np.int32) #[0 for _ in range(NUM_CLASSES)]
    total_correct_class = np.zeros(NUM_CLASSES,dtype=np.int32) #[0 for _ in range(NUM_CLASSES)]
    confusion_m=np.zeros((NUM_CLASSES*2,2),dtype=np.int32)
    
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val, sec_pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred'],ops['sec_pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)

        pred_val=np.round(sec_pred_val)
        predicted_pos=np.nonzero(pred_val[0:bsize]==0)[0]
        actual_pos=np.nonzero(batch_label[0:bsize]==0)[0]
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
        for i in range(0, bsize):
            c = cur_batch_label[i]
            l = np.nonzero(cur_batch_label[i])[0]
            t = pred_val[i]
            for k in range(c.size):
                if c[k]==1 and t[k]==1:
                    #work out index in confusion
                    confusion_m[k*2,0]+=1
                    total_correct_class[k]+=1
                if c[k]==0 and t[k]==1:
                    confusion_m[k*2+1,0]+=1
                if c[k]==0 and t[k]==0:
                    confusion_m[k*2+1,1]+=1
                if c[k]==1 and t[k]==0:
                    confusion_m[k*2,1]+=1
            total_seen_class[l] += 1
    mean_loss=loss_sum / float(batch_idx)

    log_string('eval mean loss: %f' % (mean_loss))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(total_correct_class/np.array(total_seen_class,dtype=np.float))))
    if (total_tp+total_fp)>0:
        log_string('eval precision %f'%(total_tp/float(total_tp+total_fp)))
    else:
        log_string('eval precision 0')
    if total_p>0:
        log_string('eval recall %f'%(total_tp/float(total_p)))
    else:
        log_string('eval recall 0')
    EPOCH_CNT += 1

    TEST_DATASET.reset()
    # return confusion_m and mean
    return confusion_m, mean_loss


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
    dir_name=LOG_DIR+'/AFF_'+NUM_CLASSES+'_BATCH_'+str(BATCH_SIZE)+'_DATA_'+D_TYPE
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    print('Moving results to dir %s'%dir_name)
    os.system('mv log/ %s'%(dir_name))