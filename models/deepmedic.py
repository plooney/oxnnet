import os
import json
import math
import random
import glob
import numpy as np
import tensorflow as tf
import tflearn
from oxcnn.volume_handler import VolumeSegment, ImageHandler
from oxcnn.data_loader import StandardDataLoader
from oxcnn.record_writer import RecordWriter, StandardProcessTup, RecordReader
from oxcnn.full_inferer import StandardFullInferer

segment_size_in = np.array([25]*3)
segment_size_out = segment_size_in-16
crop_by = 8
train_eval_test_no = [80,10,10]
stride = np.array([8,8,8],dtype=np.int)

def build_record_writer(data_dir, dir_type_flag):
    data_loader = StandardDataLoader(stride, segment_size_in, crop_by=crop_by)
    if dir_type_flag == 'meta':
        data_loader.read_metadata(data_dir)
    elif dir_type_flag == 'deepmedic':
        data_loader.read_deepmedic_dir(data_dir)
    else:
        data_loader.read_data_dir(data_dir, train_eval_test_no)
    return RecordWriter(data_loader, StandardProcessTup)

def build_full_inferer():
    return StandardFullInferer(segment_size_in, segment_size_out, crop_by)

class Model(object):
    def __init__(self, reuse=False, tf_record_dir=None, batch_size=0, num_epochs=0):
        record_reader = RecordReader(segment_size_in,segment_size_out)
        x_shape = [-1] + list(segment_size_in) + [1]
        y_shape = [-1] + list(segment_size_out) + [1]
        with tf.variable_scope("input") as scope:
            if tf_record_dir:
                if reuse:
                    X, Y = record_reader.input_pipeline(False, batch_size, None, tf_record_dir)
                else:
                    X, Y = record_reader.input_pipeline(True, batch_size, num_epochs, tf_record_dir)
                self.X = tf.reshape(X, x_shape)
                self.Y = tf.reshape(Y, y_shape)
            else:
                self.X = tf.placeholder(
                    dtype=tf.float32,
                    shape=[None] + segment_size_in.tolist() + [1])
                self.Y = tf.placeholder(
                    dtype=tf.float32,
                    shape=[None] + segment_size_out.tolist() + [1])
            #X = (tf.cast(self.X, tf.float32) * (2. / 255)) - 1
            X = self.X
            Y = tf.cast(self.Y, tf.float32)
        with tf.variable_scope("inference") as scope:
            if reuse:
                scope.reuse_variables()
                logits = self.build_net(X,reuse=True,scope=scope)
            else: 
                logits = self.build_net(X,reuse=False,scope=scope)
            with tf.variable_scope("pred") as scope:
                self.pred = tf.sigmoid(logits)
            with tf.variable_scope("dice") as scope:
                self.dice_op = tf.divide(2*tf.reduce_sum(tf.multiply(self.pred, Y)),
                                      tf.reduce_sum(self.pred)+tf.reduce_sum(Y),name='dice')
            with tf.variable_scope("accuracy") as scope:
                self.accuracy_op = tf.divide(tf.reduce_sum(tf.multiply(self.pred, Y)),
                                          tf.reduce_sum( tf.maximum(self.pred, Y)),name='accuracy')
            with tf.variable_scope("loss") as scope:
                self.loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y), name='cross_entropy')
        tf.summary.scalar('dice', self.dice_op)
        tf.summary.scalar('accuracy', self.accuracy_op)
        tf.summary.scalar('loss', self.loss_op)

    def build_net(self, X, reuse=False, scope=None):
        # Using TFLearn wrappers for network building
        net = tflearn.layers.conv_3d(X, 30, 3, activation='prelu',padding='valid',reuse=reuse,scope='conv1')
        net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope='batch1')
        net = tflearn.layers.conv_3d(net, 30, 3, activation='prelu',padding='valid',reuse=reuse,scope='conv2')
        net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope='batch2')
        net = tflearn.layers.conv_3d(net, 30, 3, activation='prelu',padding='valid',reuse=reuse,scope='conv3')
        net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope='batch3')
        net = tflearn.layers.conv_3d(net, 30, 3, activation='prelu',padding='valid',reuse=reuse,scope='conv4')

        net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope='batch4')
        net = tflearn.layers.conv_3d(net, 30, 3, activation='prelu',padding='valid',reuse=reuse,scope='conv5')
        net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope='batch5')
        net = tflearn.layers.conv_3d(net, 30, 3, activation='prelu',padding='valid',reuse=reuse,scope='conv6')
        net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope='batch6')
        net = tflearn.layers.conv_3d(net, 30, 3, activation='prelu',padding='valid',reuse=reuse,scope='conv7')
        net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope='batch7')
        net = tflearn.layers.conv_3d(net, 30, 3, activation='prelu',padding='valid',reuse=reuse,scope='conv8')

        net = tflearn.layers.conv_3d(net, 300, 1, activation='prelu',padding='valid',reuse=reuse,scope='fc1')
        net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope='batch8')
        net = tflearn.layers.core.dropout (net, 0.5)
        net = tflearn.layers.conv_3d(net, 300, 1, activation='prelu',padding='valid',reuse=reuse,scope='fc2')
        net = tflearn.layers.core.dropout (net, 0.5)
        net = tflearn.layers.conv_3d(net, 1, 1, activation='linear',padding='valid',reuse=reuse,scope='output')
        return net
