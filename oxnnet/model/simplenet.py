import numpy as np
import tensorflow as tf
import tflearn
from oxnnet.data_loader import StandardDataLoader
from oxnnet.record import RecordWriter, StandardProcessTup, RecordReader
from oxnnet.full_inferer import StandardFullInferer
from oxnnet.model import AbstractModel

segment_size_in = np.array([25]*3)
segment_size_in_ss = np.array([19]*3)
segment_size_out = segment_size_in-16
crop_by = 8
train_eval_test_no = [80, 10, 10]
stride = np.array([9, 9, 9], dtype=np.int)
batch_size_test = 30

data_loader = StandardDataLoader(stride, segment_size_in, crop_by=crop_by,
                                 aug_pos_samps=False, equal_class_size=True)

def build_record_writer(data_dir, dir_type_flag):
    if dir_type_flag == 'meta':
        data_loader.read_metadata(data_dir)
    elif dir_type_flag == 'deepmedic':
        data_loader.read_deepmedic_dir(data_dir)
    else:
        data_loader.read_data_dir(data_dir, train_eval_test_no)
    return RecordWriter(data_loader, StandardProcessTup)

class Model(AbstractModel):
    def __init__(self, batch_size, reuse=False, tf_record_dir=None, num_epochs=0, weighting=[1]*2):
        self.batch_size = batch_size
        record_reader = RecordReader(StandardProcessTup(data_loader))
        x_shape = [-1] + list(segment_size_in) + [1]
        y_shape = [-1] + list(segment_size_out) + [1]
        with tf.device('/cpu:0'):
            with tf.variable_scope("input"):
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
                X = self.X
                Y = tf.cast(tf.one_hot(tf.reshape(tf.cast(self.Y, tf.uint8), [-1]+list(segment_size_out)), 2), tf.float32)
        with tf.variable_scope("inference") as scope:
            if reuse:
                scope.reuse_variables()
                logits = self.build_net(X, reuse=True, scope=scope)
            else:
                logits = self.build_net(X, reuse=False, scope=scope)
            with tf.variable_scope("pred"):
                softmax_logits = tf.nn.softmax(logits)
                self.pred = tf.cast(tf.argmax(softmax_logits, axis=4), tf.float32)
            with tf.variable_scope("dice"):
                self.dice_op = tf.divide(tf.reduce_sum(tf.multiply(softmax_logits, Y)),
                                         tf.reduce_sum(self.pred) + tf.reduce_sum(Y), name='dice')
            with tf.variable_scope("loss") as scope:
                self.loss_op = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y),
                    name='cross_entropy')
                #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                #reg_constant = 0.0005  # Choose an appropriate one.
                #self.loss_op += reg_constant * sum(reg_losses)
            # Choose the metrics to compute:
            names_to_values, names_to_updates = tf.contrib.metrics.aggregate_metric_map({
                'accuracy': tf.contrib.metrics.streaming_accuracy(softmax_logits, Y),
                'precision': tf.contrib.metrics.streaming_precision(softmax_logits, Y),
                'recall': tf.contrib.metrics.streaming_recall(softmax_logits, Y),
                'mse': tf.contrib.metrics.streaming_mean_squared_error(softmax_logits, Y),
            })
            self.mse = names_to_values['mse']
            with tf.variable_scope("metrics"):
                self.metric_update_ops = list(names_to_updates.values())
            if tf_record_dir:
                tf.summary.scalar('dice', self.dice_op)
                #tf.summary.scalar('precision', self.precision_op)
                #tf.summary.scalar('recall', self.recall_op)
                #tf.summary.scalar('mse', self.mse_op)
                tf.summary.scalar('loss', self.loss_op)
            for metric_name, metric_value in names_to_values.items():
                op = tf.summary.scalar(metric_name, metric_value)

    def build_full_inferer(self):
        return StandardFullInferer(segment_size_in, segment_size_out, crop_by, stride, self.batch_size)

    def build_net(self, X, reuse=False):
        net = tflearn.layers.conv_3d(X, 30, 3, activation='relu', padding='valid', reuse=reuse, scope='conv1')
        net = tflearn.layers.conv_3d(net, 30, 3, activation='linear', padding='valid', reuse=reuse, scope='conv2')
        net = tflearn.layers.normalization.batch_normalization(net, reuse=reuse, scope='batch1')
        net = tflearn.activation(net, 'relu')

        net = tflearn.layers.conv_3d(net, 30, 3, activation='relu', padding='valid', reuse=reuse, scope='conv3')
        net = tflearn.layers.conv_3d(net, 30, 3, activation='relu', padding='valid', reuse=reuse, scope='conv4')

        net = tflearn.layers.conv_3d(net, 30, 3, activation='relu', padding='valid', reuse=reuse, scope='conv5')
        net = tflearn.layers.conv_3d(net, 30, 3, activation='relu', padding='valid', reuse=reuse, scope='conv6')
        net = tflearn.layers.conv_3d(net, 30, 3, activation='relu', padding='valid', reuse=reuse, scope='conv7')
        net = tflearn.layers.conv_3d(net, 30, 3, activation='relu', padding='valid', reuse=reuse, scope='conv8')

        net = tflearn.layers.conv_3d(net, 300, 1, activation='relu', padding='valid', reuse=reuse, scope='fc1')
        net = tflearn.layers.core.dropout(net, 0.5)
        net = tflearn.layers.conv_3d(net, 300, 1, activation='relu', padding='valid', reuse=reuse, scope='fc2')
        net = tflearn.layers.core.dropout(net, 0.5)
        net = tflearn.layers.conv_3d(net, 2, 1, activation='linear', padding='valid', reuse=reuse, scope='output')
        return net
