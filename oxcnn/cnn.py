import argparse
import os
import sys
import tensorflow as tf
import tflearn
import tflearn.helpers.summarizer as s
import numpy as np
import json
import math
import time
from datetime import datetime, timedelta
import pandas as pd
import importlib
import models

class CNN(object):

    def write_to_csv(file_name):
        pd.DataFrame(data=train_loss_iterations,
                     columns=list(train_loss_iterations.keys())
                    ).to_csv(os.path.join(save_dir, 'log.csv'))
        pd.DataFrame(data=full_validation_metrics,
                     columns=list(full_validation_metrics.keys())
                    ).to_csv(os.path.join(save_dir, 'full_validation_log.csv'))

    def get_testdata(self, meta_data_file):
        with open(meta_data_file,'r') as f: d = json.load(f)
        return d['test_tups']

    def train(self, tf_record_dir, save_dir, test_data, module, num_epochs, batch_size, num_save_every, model_file=None):
        train_loss_iterations = {'iteration': [], 'epoch': [], 'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': []}
        meta_data_filepath = os.path.join(tf_record_dir,'meta_data.txt')
        with open(meta_data_filepath,'r') as f:
            meta_data = json.load(f)
        num_examples = sum([x[1] for x in meta_data['train_examples'].items()])
        num_batches_per_epoch = num_examples//batch_size
        num_batches = math.ceil(num_epochs*num_batches_per_epoch)
        num_examples_val = sum([x[1] for x in meta_data['validation_examples'].items()])
        num_batches_per_epoch_val = num_examples_val/batch_size
        num_full_validation_every = int(1 * num_batches_per_epoch)
        save_pred_dir=os.path.join(os.getcwd(),'save_preds')
        validation_tups = meta_data['validation_tups']
        full_validation_metrics = {k[0]:[] for  k in validation_tups}

        with tf.Graph().as_default():
            tflearn.config.init_training_mode()
            with tf.name_scope('training') as scope:
                model = module.Model(batch_size, False, tf_record_dir, num_epochs,'training')
                tf.summary.scalar('dice', model.dice_op)
                tf.summary.scalar('accuracy', model.accuracy_op)
                tf.summary.scalar('loss', model.loss_op)
            with tf.name_scope('eval'):
                model_eval = module.Model(batch_size, True, tf_record_dir, num_epochs, 'training') if num_save_every else None
                tf.summary.scalar('dice', model_eval.dice_op)
                tf.summary.scalar('accuracy', model_eval.accuracy_op)
                tf.summary.scalar('loss', model_eval.loss_op)
            model_test = module.Model(batch_size, True, scope='training')
            inferer = model_test.build_full_inferer()
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer().minimize(model.loss_op, global_step=global_step, aggregation_method = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            avg_time = 0
            with tf.Session(config = config) as sess:
            #with tf.Session() as sess:
                merged = s.summarize_variables()
                merged = tf.summary.merge_all()
                #merged_train = tf.summary.merge(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='training'))
                #merged_eval = tf.summary.merge(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval'))
                train_writer = tf.summary.FileWriter(save_dir + '/train',sess.graph)
                eval_writer = tf.summary.FileWriter(save_dir + '/test')
                sess.run(tf.local_variables_initializer())
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                coord = tf.train.Coordinator()
                tf.train.start_queue_runners(sess, coord=coord)
                if model_file:
                    saver.restore(sess, model_file)
                try:
                    while not coord.should_stop():
                        tflearn.is_training(True)
                        cur_step = sess.run(global_step)
                        epoch = cur_step//num_batches_per_epoch
                        start = time.time()
                        #print(tf.get_collection('summaries'))
                        train_loss, train_dice, train_accuracy, _, summaries = sess.run([model.loss_op, model.dice_op, model.accuracy_op, optimizer, merged])
                        #summaries = sess.run(merged)
                        #train_loss, train_dice, train_accuracy, _ = sess.run([model.loss_op, model.dice_op, model.accuracy_op, optimizer])
                        end = time.time()
                        avg_time = avg_time + ((end-start)-avg_time)/cur_step if cur_step else end - start
                        time_remaining_s = (num_batches - cur_step)*avg_time
                        t = timedelta(seconds=time_remaining_s);
                        time_remaining_string = "time left: {}m-{}d {} (h:mm:ss)".format(t.days/30, t.days%30, timedelta(seconds=t.seconds))
                        print("{}/{} (epoch {}), train_loss = {:.3f}, dice = {:.3f}, accuracy = {:.3f}, time/batch = {:.3f}, " \
                              .format(cur_step,num_batches,
                                      epoch, train_loss, train_dice, train_accuracy, end - start) + time_remaining_string  )
                        train_loss_iterations['iteration'].append(cur_step)
                        train_loss_iterations['epoch'].append(epoch)
                        train_loss_iterations['train_loss'].append(train_loss)
                        train_loss_iterations['train_dice'].append(train_dice)
                        #if num_batches_val and cur_step % num_save_every == 0:
                        if cur_step % num_save_every == 0:
                            tflearn.is_training(False)
                            start = time.time()
                            train_writer.add_summary(summaries, cur_step)
                            val_loss, val_dice, val_accuracy, summaries = sess.run([model_eval.loss_op, model_eval.dice_op, model_eval.accuracy_op, merged])
                            eval_writer.add_summary(summaries, cur_step)
                            print('val_loss: {:.3f}, val_dice: {:.3f}, accuracy: {:.3f}'.format(val_loss,val_dice,val_accuracy))
                            train_loss_iterations['val_loss'].append(val_loss)
                            train_loss_iterations['val_dice'].append(val_dice)
                            checkpoint_path = os.path.join(save_dir, 'model.ckpt')
                            saver.save(sess, checkpoint_path, global_step=global_step)
                            print(saver.last_checkpoints)
                            print("model saved to {}".format(checkpoint_path))
                            end = time.time()
                            avg_time = avg_time + ((end-start)-avg_time)/cur_step if cur_step else avg_time  
                        else:
                            train_loss_iterations['val_loss'].append(None)
                            train_loss_iterations['val_dice'].append(None)

                        if num_full_validation_every and cur_step % num_full_validation_every == 0 and cur_step:
                            tflearn.is_training(False)
                            start = time.time()
                            test_save_dir = os.path.join(save_dir,'val_preds_' + str(cur_step))
                            for tup in validation_tups:
                                dice = inferer(sess, tup, save_dir, model_test)
                                full_validation_metrics[tup[0]].append(dice)
                                print(dice)
                            end = time.time()
                            avg_time = avg_time + ((end-start)/num_full_validation_every-avg_time)/cur_step if cur_step else avg_time  
                except tf.errors.OutOfRangeError as e :
                    print('Done training')
                finally:
                    checkpoint_path = os.path.join(save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=global_step)
                    coord.request_stop()
                print('Finished')

    def test(self, save_dir, test_data, model_file, module):
        with tf.get_default_graph().as_default():
            #config = tf.ConfigProto()
            #config.gpu_options.allow_growth=True
            avg_time = 0
            model = module.Model(batch_size, False) #get_model_with_placeholders(module, reuse=False)
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
                    inferer = model.build_full_inferer()
                    dice = inferer(sess, tup, save_dir, model)
                    dices[tup[0]] = dice
                    print(dice)
                print(dices)

    def write_records(self, output_dir, data_dir, dir_type_flag, module):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        if os.listdir(output_dir): print("Warning: Directory not empty - metadata may be incorrect it directory contains TFRecords")
        module.build_record_writer(data_dir, dir_type_flag).write_records(output_dir)
