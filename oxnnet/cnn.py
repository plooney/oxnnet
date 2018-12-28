"""Module with one class CNN"""
import os
import json
import math
import time
from datetime import timedelta
import tflearn
import tflearn.helpers.summarizer as s
import numpy as np
import pandas as pd
import tensorflow as tf

def write_csv(fname, my_dict, mode='w', header=True):
    """Writes (appends or with header) csv of validation metrics"""
    pd.DataFrame.from_dict(my_dict).to_csv(fname, mode=mode, header=header)
    
def get_testdata(meta_data_file):
    """Reads the image files for testing from the metadata file in the records dir."""
    with open(meta_data_file, 'r') as f: d = json.load(f)
    return d['test_tups']

class CNN(object):
    """Class to perform writing of records, training, testing and feature writing"""
    def __init__(self, module):
        self.module = module

    def train(self, tf_record_dir, save_dir, num_epochs, batch_size, num_save_every,
              model_file=None, early_stop=False, full_eval_every=0, learning_rate=1e-3, lr_steps=0, lr_decay=0.96, avg = False):
        """Trains the Model defined in module on the records in tf_record_dir"""
        train_loss_iterations = {'iteration': [], 'epoch': [], 'train_loss': [], 'train_dice': [],
                                 'train_mse': [], 'val_loss': [], 'val_dice': [], 'val_mse': []}
        meta_data_filepath = os.path.join(tf_record_dir, 'meta_data.txt')
        with open(meta_data_filepath, 'r') as f:
            meta_data = json.load(f)
        num_examples = sum([x[1] for x in meta_data['train_examples'].items()])
        num_batches_per_epoch = num_examples//batch_size
        num_batches = math.ceil(num_epochs*num_batches_per_epoch) if num_epochs else 0
        num_full_validation_every = full_eval_every if full_eval_every else num_batches_per_epoch
        validation_tups = meta_data['validation_tups']
        full_validation_metrics = {k[0]:[] for  k in validation_tups}
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            write_csv(os.path.join(save_dir, 'log.csv'), train_loss_iterations)
            write_csv(os.path.join(save_dir, 'full_validation_log.csv'), full_validation_metrics)
        with tf.Graph().as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            tflearn.config.init_training_mode()
            with tf.name_scope('training'):
                model = self.module.Model(batch_size, False, tf_record_dir, num_epochs)
            with tf.name_scope('eval'):
                model_eval = self.module.Model(batch_size, True, tf_record_dir,
                                               num_epochs) if validation_tups else None
            model_test = self.module.Model(batch_size, True)
            inferer = model_test.build_full_inferer()
            avg_time = 0
            global_step = tf.Variable(0, name='global_step', trainable=False)
            lr = tf.train.exponential_decay(learning_rate, global_step, lr_steps, lr_decay, staircase=True) if lr_steps else learning_rate

            optimizer = tf.train.AdamOptimizer(lr)
            tf.summary.scalar('learning_rate', lr)
            update_op = self._avg(model.loss_op, optimizer, global_step) if avg else optimizer.minimize(model.loss_op,
                                                                                                        global_step=global_step)
            #update_op = optimizer.minimize(model.loss_op,
            #                               global_step=global_step)

            #config = tf.ConfigProto()
            #config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                merged = s.summarize_variables()
                merged = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(save_dir, sess.graph)
                sess.run(tf.local_variables_initializer())
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver_epoch = tf.train.Saver(max_to_keep=None)
                coord = tf.train.Coordinator()
                if model_file:
                    vars_to_restore = model.filter_vars(tf.global_variables())
                    restore_saver = (tf.train.Saver(vars_to_restore)
                                     if vars_to_restore else tf.train.Saver())
                    restore_saver.restore(sess, model_file)
                try:
                    continue_training = True
                    epoch = previous_epoch = 0
                    while continue_training and not coord.should_stop():
                        tflearn.is_training(True)
                        cur_step = sess.run(global_step)
                        previous_epoch = epoch
                        epoch = cur_step//num_batches_per_epoch
                        start = time.time()
                        if epoch != previous_epoch: saver = tf.train.Saver()
                        #print(tf.get_collection('summaries'))
                        #if num_batches_val and cur_step % num_save_every == 0:
                        message_string = ''
                        if num_save_every and cur_step % num_save_every == 0:
                            tflearn.is_training(False)
                            start = time.time()
                            (train_loss, train_dice, val_loss, val_dice, summaries,
                             train_mse, val_mse) = sess.run([model.loss_op,
                                                             model.dice_op,
                                                             model_eval.loss_op,
                                                             model_eval.dice_op,
                                                             merged, model.mse,
                                                             model_eval.mse,
                                                             ] +
                                                            model_eval.metric_update_ops +
                                                            model.metric_update_ops)[0:7]
                            summary_writer.add_summary(summaries, cur_step)
                            message_string = (' val_loss: {:.3f}, val_dice: {:.3f}, mse: {:.5f}'
                                              .format(val_loss, val_dice, val_mse))
                            non_zero_losses = [x for x in  train_loss_iterations['val_loss']
                                               if x is not None]
                            non_zero_dices = [x for x in  train_loss_iterations['val_dice']
                                              if x is not None]
                            non_zero_val_mses = [x for x in  train_loss_iterations['val_mse']
                                                 if x is not None and x > 0]
                            should_save = (val_loss < np.percentile(non_zero_losses, 25)
                                           if non_zero_losses else True)
                            should_save = (should_save and val_dice > np.percentile(non_zero_dices, 75)
                                           if non_zero_dices else True)
                            min_mse = 0 if not non_zero_val_mses else np.min(non_zero_val_mses)
                            should_save = should_save and val_mse < min_mse and min_mse
                            train_loss_iterations['val_loss'].append(val_loss)
                            train_loss_iterations['val_dice'].append(val_dice)
                            train_loss_iterations['val_mse'].append(val_mse)
                            if should_save:
                                checkpoint_path = os.path.join(save_dir, 'epoch_'
                                                               + str(epoch) + '_model.ckpt')
                                saver.save(sess, checkpoint_path, global_step=global_step)
                                print("model saved to {}".format(checkpoint_path))
                            non_zero_val_mses = [x for x in  train_loss_iterations['val_mse']
                                                 if x is not None and x > 0]
                            if early_stop:
                                continue_training = (np.median(non_zero_val_mses[-10:])
                                                     <= np.min(non_zero_val_mses[:-10])
                                                     if non_zero_val_mses[:-10] and epoch > 0
                                                     else True)
                        else:
                            train_loss_iterations['val_loss'].append(None)
                            train_loss_iterations['val_dice'].append(None)
                            train_loss_iterations['val_mse'].append(None)
                            train_loss, train_dice, train_mse = sess.run([model.loss_op,
                                                                          model.dice_op,
                                                                          model.mse
                                                                          ])[0:3]
                        train_loss_iterations['iteration'].append(cur_step)
                        train_loss_iterations['epoch'].append(epoch)
                        train_loss_iterations['train_loss'].append(train_loss)
                        train_loss_iterations['train_dice'].append(train_dice)
                        train_loss_iterations['train_mse'].append(train_mse)
                        write_csv(os.path.join(save_dir, 'log.csv'),
                                  {k:([v[-1]] if v else [])
                                   for k, v in train_loss_iterations.items()},
                                  mode='a', header=False)
                        end = time.time()
                        avg_time = avg_time + ((end-start)-avg_time)/cur_step if cur_step else end - start
                        time_remaining_s = (num_batches - cur_step)*avg_time if num_epochs else 0
                        t = timedelta(seconds=time_remaining_s)
                        time_remaining_string = ("time left: {}m-{}d {} (h:mm:ss)"
                                                 .format(t.days/30, t.days%30,
                                                         timedelta(seconds=t.seconds)))
                        message_string = ("{}/{} (epoch {}), train_loss = {:.3f}, dice = {:.3f}, accuracy = {:.3f}, time/batch = {:.3f}, "
                                          .format(cur_step, num_batches, epoch, train_loss, train_dice,
                                                  0, end - start) + time_remaining_string +
                                          message_string)
                        print(message_string)
                        if all([num_full_validation_every,
                                cur_step % num_full_validation_every == 0,
                                cur_step]):
                            tflearn.is_training(False)
                            start = time.time()
                            test_save_dir = os.path.join(save_dir, 'val_preds_' + str(cur_step))
                            if not os.path.exists(test_save_dir): os.makedirs(test_save_dir)
                            for tup in validation_tups:
                                dice = inferer(sess, tup, test_save_dir, model_test)
                                full_validation_metrics[tup[0]].append(dice)
                                print(dice)
                            end = time.time()
                            avg_time = avg_time + ((end-start)/num_full_validation_every
                                                   - avg_time)/cur_step if cur_step else avg_time
                            checkpoint_path = os.path.join(save_dir, 'epoch_model.ckpt')
                            saver_epoch.save(sess, checkpoint_path, global_step=global_step)
                            print("epoch model saved to {}".format(checkpoint_path))
                            write_csv(os.path.join(save_dir, 'full_validation_log.csv'),
                                      {k:([v[-1]] if v else [])
                                       for k, v in full_validation_metrics.items()},
                                      mode='a',
                                      header=False)
                        #Run optimiser after saving/evaluating the model 
                        sess.run(update_op)
                except tf.errors.OutOfRangeError:
                    print('Done training')
                finally:
                    checkpoint_path = os.path.join(save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=global_step)
                    coord.request_stop()
                print('Finished')

    def test(self, save_dir, test_data, model_file, batch_size, avg=False):
        #with tf.get_default_graph().as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            model = self.module.Model(batch_size, False) #get_model_with_placeholders(self.module, reuse=False)
            if avg:
                variable_averages = tf.train.ExponentialMovingAverage(0.999)
                variables_to_restore = variable_averages.variables_to_restore()
                saver = tf.train.Saver(variables_to_restore)
            else:
                saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, model_file)
                tflearn.is_training(False)
                with tf.variable_scope("inference") as scope:
                    scope.reuse_variables()
                scope.reuse_variables()
                print(sess.run(tflearn.get_training_mode()))
                dices = {}
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                for tup in test_data:
                    inferer = model.build_full_inferer()
                    dice = inferer(sess, tup, save_dir, model)
                    dices[tup[0]] = dice
                    print("Median: {}, Max: {}, Min: {}"
                          .format(np.median(list(dices.values()), axis=0),
                                  np.max(list(dices.values()), axis=0),
                                  np.min(list(dices.values()), axis=0)))
                print(dices)
                return(dices)
                sess.close()

    def feats(self, save_dir, test_data, model_file, batch_size):
        with tf.get_default_graph().as_default():
            #config = tf.ConfigProto()
            #config.gpu_options.allow_growth=True
            model = self.module.Model(batch_size, False) #get_model_with_placeholders(self.module, reuse=False)
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, model_file)
                tflearn.is_training(False)
                #with tf.variable_scope("inference") as scope:
                #    scope.reuse_variables()
                    #print(sess.run(tf.get_variable('level7/trans4/W')))
                #scope.reuse_variables()
                print(sess.run(tflearn.get_training_mode()))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                for tup in test_data:
                    feats_writer = model.build_feats_writer()
                    feats_writer(sess, tup, save_dir, model)

    def write_records(self, output_dir, data_dir, dir_type_flag):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if os.listdir(output_dir):
            print("Warning: Directory not empty - metadata may be incorrect it directory contains TFRecords")
        self.module.build_record_writer(data_dir, dir_type_flag).write_records(output_dir)

    def _add_loss_summaries(self, total_loss):
        """Add summaries for losses in CIFAR-10 model.
        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.
        Args:
            total_loss: Total loss from loss().
        Returns:
            loss_averages_op: op for generating moving averages of losses.
            """
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.999, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.summary.scalar(l.op.name + ' (raw)', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))

        return loss_averages_op

    def _avg(self, loss_op, opt, global_step):
        print("Doing avg")
        # GeneraVte moving averages of all losses and associated summaries.
        loss_averages_op = self._add_loss_summaries(loss_op)

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            #opt = tf.train.GradientDescentOptimizer(0.001)
            grads = opt.compute_gradients(loss_op)

            # Apply gradients.
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', grad)

            # Track the moving averages of all trainable variables.
            variable_averages = tf.train.ExponentialMovingAverage(
                0.999, global_step)
            with tf.control_dependencies([apply_gradient_op]):
                variables_averages_op = variable_averages.apply(tf.trainable_variables())

            return variables_averages_op
            #return loss_op
        #return opt.minimize(loss_op,
        #                    global_step=global_step)
