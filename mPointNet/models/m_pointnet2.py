"""
    PointNet++ Model for point clouds classification
"""

import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
# import numpy as np
import tf_util
from m_pointnet_util import pointnet_sa_module


def new_placeholder_inputs(batch_size,num_point,n_classes=2):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size,n_classes))
    return pointclouds_pl,labels_pl


def new_get_model(point_cloud, is_training, bn_decay=None,n_classes=2,wdecay=0.1):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = None
    end_points['l0_xyz'] = l0_xyz

    # Set abstraction layers
    # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
    # So we only use NCHW for layer 1 until this issue can be resolved.
    l1_xyz, l1_points, l1_indices, l1_subindices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', use_nchw=True, wd=wdecay)
    l2_xyz, l2_points, l2_indices , _ = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2', wd=wdecay)
    l3_xyz, l3_points, l3_indices, _ = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3', wd=wdecay)

    # Fully connected layers
    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay,weight_decay=wdecay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay,weight_decay=wdecay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, n_classes, activation_fn=None, scope='fc3',weight_decay=wdecay)
    sigmoid_tensor = tf.sigmoid(net,name='final_result')
    # l1_xyz = tf.Print(l1_xyz, [tf.shape(l1_xyz)], "XYZ", summarize=5)
    # l1_indices = tf.Print(l1_indices, [tf.shape(l1_indices)], "Indices", summarize=5)
    # l1_subindices = tf.Print(l1_subindices, [tf.shape(l1_subindices)], "SubIds", summarize=5)
    return net, sigmoid_tensor, end_points, l1_xyz, l1_indices, l1_subindices


def new_get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    #loss1=--tf.reduce_sum( (  (y_*tf.log(out + 1e-9)) + ((1-y_) * tf.log(1 - out + 1e-9)) )  , name='xentropy' ) 
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)

    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss    
