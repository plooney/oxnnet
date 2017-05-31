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

    def write_csv(self,fname, my_dict, mode='w', header=True):
        pd.DataFrame(data=my_dict,
                     columns=list(my_dict.keys())
                    ).to_csv(fname, mode=mode, header=header)

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
        num_batches = math.ceil(num_epochs*num_batches_per_epoch) if num_epochs else 0
        num_examples_val = sum([x[1] for x in meta_data['validation_examples'].items()]) if num_epochs else 0
        num_batches_per_epoch_val = num_examples_val/batch_size
        num_full_validation_every = int(1 * num_batches_per_epoch)
        save_pred_dir=os.path.join(os.getcwd(),'save_preds')
        validation_tups = meta_data['validation_tups']
        full_validation_metrics = {k[0]:[] for  k in validation_tups}
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            self.write_csv(os.path.join(save_dir, 'log.csv'), train_loss_iterations)
            self.write_csv(os.path.join(save_dir, 'full_validation_log.csv'), full_validation_metrics)

        with tf.Graph().as_default():
            tflearn.config.init_training_mode()
            with tf.name_scope('training') as scope:
                model = module.Model(batch_size, False, tf_record_dir, num_epochs)
            with tf.name_scope('eval'):
                model_eval = module.Model(batch_size, True, tf_record_dir, num_epochs) if num_save_every else None
            model_test = module.Model(batch_size, True)
            inferer = model_test.build_full_inferer()
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer().minimize(model.loss_op, global_step=global_step)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            avg_time = 0
            with tf.Session(config = config) as sess:
            #with tf.Session() as sess:
                merged = s.summarize_variables()
                merged = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(save_dir,sess.graph)
                sess.run(tf.local_variables_initializer())
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver_epoch = tf.train.Saver(max_to_keep=None)
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
                        #if num_batches_val and cur_step % num_save_every == 0:
                        message_string = '' 
                        if cur_step % num_save_every == 0:
                            tflearn.is_training(False)
                            start = time.time()
                            train_loss, train_dice, val_loss, val_dice, summaries = sess.run([model.loss_op, model.dice_op, model_eval.loss_op, model_eval.dice_op, merged, optimizer] + model_eval.metric_update_ops + model.metric_update_ops)[0:5]
                            summary_writer.add_summary(summaries, cur_step)
                            message_string = ' val_loss: {:.3f}, val_dice: {:.3f}, mse: {:.3f}'.format(val_loss,val_dice, 0)
                            train_loss_iterations['val_loss'].append(val_loss)
                            train_loss_iterations['val_dice'].append(val_dice)
                            should_save = val_loss < np.median(sorted([x for x in  train_loss_iterations['val_loss'][:-10] if x is not None])[0:10])
                            should_save = should_save and np.median([x for x in  train_loss_iterations['val_loss'] if x is not None][-10:]) < np.median(sorted([x for x in  train_loss_iterations['val_loss'] if x is not None])[:-10])
                            #if not train_loss_iterations['val_loss'] or should_save:
                            #    checkpoint_path = os.path.join(save_dir, 'model.ckpt')
                            #    saver.save(sess, checkpoint_path, global_step=global_step)
                            #    print("model saved to {}".format(checkpoint_path))
                        else:
                            train_loss_iterations['val_loss'].append(None)
                            train_loss_iterations['val_dice'].append(None)
                            train_loss, train_dice, _ = sess.run([model.loss_op, model.dice_op, optimizer])[0:3]
                        train_loss_iterations['iteration'].append(cur_step)
                        train_loss_iterations['epoch'].append(epoch)
                        train_loss_iterations['train_loss'].append(train_loss)
                        train_loss_iterations['train_dice'].append(train_dice)
                        self.write_csv(os.path.join(save_dir, 'log.csv'), 
                                  {k:([v[-1]] if v else []) for k,v in train_loss_iterations.items()}, 
                                  mode='a', header=False)
                        end = time.time()
                        avg_time = avg_time + ((end-start)-avg_time)/cur_step if cur_step else end - start
                        time_remaining_s = (num_batches - cur_step)*avg_time if num_epochs else 0
                        t = timedelta(seconds=time_remaining_s)
                        time_remaining_string = "time left: {}m-{}d {} (h:mm:ss)".format(t.days/30, t.days%30, timedelta(seconds=t.seconds))
                        message_string = "{}/{} (epoch {}), train_loss = {:.3f}, dice = {:.3f}, accuracy = {:.3f}, time/batch = {:.3f}, " \
                                .format(cur_step,num_batches,
                                      epoch, train_loss, train_dice, 0, end - start) + time_remaining_string + message_string 
                        print(message_string)

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
                            checkpoint_path = os.path.join(save_dir, 'epoch_model.ckpt')
                            saver.save(sess, checkpoint_path, global_step=global_step)
                            print("epoch model saved to {}".format(checkpoint_path))
                            self.write_csv(os.path.join(save_dir, 'full_validation_log.csv'),
                                      {k:([v[-1]] if v else []) for k,v in full_validation_metrics.items()},
                                      mode='a',
                                      header=False)
                except tf.errors.OutOfRangeError as e :
                    print('Done training')
                finally:
                    checkpoint_path = os.path.join(save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=global_step)
                    coord.request_stop()
                print('Finished')

    def test(self, save_dir, test_data, model_file, module, batch_size):
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
