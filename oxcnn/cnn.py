import argparse
import os
import sys
import tensorflow as tf
import tflearn
import numpy as np
import json
import math
import time
from datetime import datetime, timedelta
import pandas as pd
import importlib
import models

class CNN(object):
    #def get_model(self, module, num_epochs, tf_record_dir, batch_size, reuse):
    #    return module.Model(reuse=reuse, batch_size=batch_size, num_epochs=num_epochs, tf_record_dir=tf_record_dir)

    #def get_model_with_placeholders(self, module, reuse=False):
    #    return module.Model(reuse=reuse)

    def get_testdata(self, meta_data_file):
        with open(meta_data_file,'r') as f: d = json.load(f)
        return d['test_tups']

    def train(self, tf_record_dir, save_dir, test_data, module, num_epochs, batch_size, num_save_every, num_batches_val ):
        train_loss_iterations = {'iteration': [], 'epoch': [], 'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': []}
        #num_epochs = 2 
        #batch_size = 12
        print('here',tf_record_dir)
        meta_data_filepath = os.path.join(tf_record_dir,'meta_data.txt')
        with open(meta_data_filepath,'r') as f:
            meta_data = json.load(f)
        num_examples = sum([x[1] for x in meta_data['train_examples'].items()])
        num_batches_per_epoch = num_examples/batch_size
        num_batches = math.ceil(num_epochs*num_batches_per_epoch)
        num_examples_val = sum([x[1] for x in meta_data['validation_examples'].items()])
        num_batches_per_epoch_val = num_examples_val/batch_size
        num_batches_val = 0 #math.ceil(num_epochs*num_batches_per_epoch_val)
        num_save_every = 100
        num_full_validation_every = 0 #int(1 * num_batches_per_epoch)
        save_pred_dir=os.path.join(os.getcwd(),'save_preds')
        validation_tups = meta_data['validation_tups']
        full_validation_metrics = {k[0]:[] for  k in validation_tups}

        with tf.Graph().as_default():
            tflearn.config.init_training_mode()
            model = module.Model(False, tf_record_dir, batch_size, num_epochs)
            model_eval = module.Model(True, tf_record_dir, batch_size, num_epochs) if num_batches_val else None
            model_test = module.Model(True)
            inferer = module.build_full_inferer()
        
            optimizer = tf.train.AdamOptimizer().minimize(model.loss_op)
            # Initializing the variables
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            avg_time = 0
            #with tf.Session(config = config) as sess:
            with tf.Session() as sess:
                merged = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(save_dir + '/train',sess.graph)
                sess.run(tf.local_variables_initializer())
                sess.run(tf.global_variables_initializer())
                coord = tf.train.Coordinator()
                tf.train.start_queue_runners(sess, coord=coord)
                saver = tf.train.Saver()
                try:
                    batch_idx = 0
                    while not coord.should_stop():
                        tflearn.is_training(True)
                        epoch = batch_idx//num_batches_per_epoch
                        start = time.time()
                        train_loss, train_dice, train_accuracy, _ = sess.run([model.loss_op, model.dice_op, model.accuracy_op, optimizer])
                        end = time.time()
                        avg_time = avg_time + ((end-start)-avg_time)/batch_idx if batch_idx else end - start
                        time_remaining_s = (num_batches - batch_idx)*avg_time
                        t = timedelta(seconds=time_remaining_s);
                        time_remaining_string = "time left: {}m-{}d {} (h:mm:ss)".format(t.days/30, t.days%30, timedelta(seconds=t.seconds))
                        print("{}/{} (epoch {}), train_loss = {:.3f}, dice = {:.3f}, accuracy = {:.3f}, time/batch = {:.3f}, " \
                              .format(batch_idx,num_batches,
                                      epoch, train_loss, train_dice, train_accuracy, end - start) + time_remaining_string  )
                        train_loss_iterations['iteration'].append(batch_idx)
                        train_loss_iterations['epoch'].append(epoch)
                        train_loss_iterations['train_loss'].append(train_loss)
                        train_loss_iterations['train_dice'].append(train_dice)
                        if batch_idx % num_save_every == 0 and num_batches_val:
                            start = time.time()
                            tflearn.is_training(False)
                            avg_val_loss = 0
                            avg_val_dice = 0
                            for i in range(0,num_batches_val):
                                print(i,num_batches_val)
                                # evaluate
                                val_loss, val_dice, val_accuracy = sess.run([model_eval.loss_op, model_eval.dice_op, model_eval.accuracy_op])
                                avg_val_loss += val_loss / num_batches_val
                                avg_val_dice += val_dice / num_batches_val
                            print('val_loss: {:.3f}, val_dice: {:.3f}, accuracy: {:.3f}'.format(avg_val_loss,avg_val_dice,val_accuracy))
                            train_loss_iterations['val_loss'].append(avg_val_loss)
                            train_loss_iterations['val_dice'].append(avg_val_dice)
                            checkpoint_path = os.path.join(save_dir, 'model.ckpt')
                            saver.save(sess, checkpoint_path, global_step=batch_idx)
                            print("model saved to {}".format(checkpoint_path))
                            end = time.time()
                            avg_time = avg_time + ((end-start)/num_save_every-avg_time)/batch_idx if batch_idx else avg_time  
                        else:
                            train_loss_iterations['val_loss'].append(None)
                            train_loss_iterations['val_dice'].append(None)

                        if num_full_validation_every and batch_idx % num_full_validation_every == 0 and batch_idx:
                            tflearn.is_training(False)
                            start = time.time()
                            test_save_dir = os.path.join(save_dir,'val_preds_' + str(batch_idx))
                            for tup in validation_tups:
                                dice = inferer(sess, tup, save_dir, model_test)
                                full_validation_metrics[tup[0]].append(dice)
                                print(dice)
                            end = time.time()
                            avg_time = avg_time + ((end-start)/num_full_validation_every-avg_time)/batch_idx if batch_idx else avg_time  
                        batch_idx += 1
                except tf.errors.OutOfRangeError as e :
                    print('Done training')
                finally:
                    checkpoint_path = os.path.join(save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=batch_idx)
                    coord.request_stop()
                pd.DataFrame(data=train_loss_iterations,
                             columns=list(train_loss_iterations.keys())
                            ).to_csv(os.path.join(save_dir, 'log.csv'))
                pd.DataFrame(data=full_validation_metrics,
                             columns=list(full_validation_metrics.keys())
                            ).to_csv(os.path.join(save_dir, 'full_validation_log.csv'))
                print('Finished')

    def test(self, save_dir, test_data, model_file, module):
        with tf.get_default_graph().as_default():
            #config = tf.ConfigProto()
            #config.gpu_options.allow_growth=True
            avg_time = 0
            model = module.Model(False) #get_model_with_placeholders(module, reuse=False)
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(model_file)))
                tflearn.is_training(False)
                with tf.variable_scope("inference") as scope:
                    scope.reuse_variables()
                    #print(np.sum(sess.run(tf.get_variable('conv1/W'))))
                scope.reuse_variables()
                print(sess.run(tflearn.get_training_mode()))
                dices = {}
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                for tup in test_data:
                    inferer = get_full_inferer(module)
                    dice = inferer(sess, tup, save_dir, model)
                    dices[tup[0]] = dice
                    print(dice)
                print(dices)

    def write_records(self, output_dir, data_dir, dir_type_flag, module):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        if os.listdir(output_dir): print("Warning: Directory not empty - metadata may be incorrect it directory contains TFRecords")
        module.build_record_writer(data_dir, dir_type_flag).write_records(output_dir)

FLAGS = None

def main(unused_argv):
    print(FLAGS,unused_argv)
    modules = {
        'deepmedic':importlib.import_module('models.deepmedic'),
        'unet':importlib.import_module('models.unet'),
        'resunet':importlib.import_module('models.resunet'),
    #    'resmed':importlib.import_module('models.resmed'),
    #    'dm_deepmedic':importlib.import_module('models.dm_deepmedic'),
        'mp_deepmedic':importlib.import_module('models.mp_deepmedic'),
        'mp_deepmedic_largefov':importlib.import_module('models.mp_deepmedic_largefov'),
    }
    module = importlib.import_module(FLAGS.model[0]) #modules[FLAGS.model[0]]
    cnn = CNN()
    if FLAGS.write_tf_records:
        cnn.write_records(FLAGS.write_tf_records[0], FLAGS.write_tf_records[1], FLAGS.dir_type, module)
    if FLAGS.train:
        cnn.train(FLAGS.train[0], FLAGS.train[1], get_testdata(FLAGS.train[2]), module)
    if FLAGS.test:
        cnn.test(FLAGS.test[0], get_testdata(FLAGS.test[1]), FLAGS.test[2], module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        nargs=1,
        type=str,
        action='store')
    subparsers = parser.add_subparsers(help='sub-command help')
    parser_train = subparsers.add_parser('--train', help='train help')
    parser_train.add_argument('--tfr_dir', type=str)
    parser_train.add_argument('--save_dir', type=str)
    parser_train.add_argument('--test_data', type=str)
    #parser_train.add_argument(
    #    '--train',
    #    type=str,
    #    nargs=3,
    #    help='train help',
    #    action='store')
    parser.add_argument(
        '--test',
        nargs=3,
        action='store')
    parser.add_argument(
        '--write_tf_records',
        nargs=2,
        action='store')
    parser.add_argument(
        '--dir_type',
        nargs='?',
        default = '',
        type=str,
        action='store')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
