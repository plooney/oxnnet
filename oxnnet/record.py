"""This modules contains classes to enable writing and reading tensorflow records"""
import os
import json
from multiprocessing import Pool
import glob
import abc
import numpy as np
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

class AbstractProcessTup(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def convert_to(self, *args):
        raise NotImplementedError('users must define conver_to to use this base class')

    def __init__(self, data_loader, prefix=None, output_dir=None):
        self.data_loader = data_loader
        self.prefix = prefix
        self.output_dir = output_dir

    def __call__(self, tup):
        name = os.path.basename(tup[0].split('.')[0])
        vals = self.data_loader.get_batch(tup)
        self.convert_to(*vals[:-1], self.prefix + '_' + name)
        return (name, len(vals[0])), (name, vals[-1])

class StandardProcessTup(AbstractProcessTup):

    def features_encode(self, label_raw, volume_raw, rows, cols, depth):
        features = tf.train.Features(
            feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'volume_seg': _bytes_feature(label_raw),
                'volume_raw': _floats_feature(volume_raw)
            }
        )
        return features

    def features_decode(self, serialized_example):
        example = tf.parse_single_example(
            serialized_example,
            features={
                'volume_raw': tf.FixedLenFeature([np.prod(self.data_loader.segment_size)], tf.float32),
                'volume_seg': tf.FixedLenFeature([], tf.string),
            }
        )
        volume = example['volume_raw']
        volume.set_shape([np.prod(self.data_loader.segment_size)])

        label = tf.decode_raw(example['volume_seg'], tf.uint8)
        label.set_shape([np.prod(self.data_loader.segment_size-2*self.data_loader.crop_by)])
        return volume, label

    def convert_to(self, *vals):
        volumes, labels, name = vals
        rows = volumes.shape[1]
        cols = volumes.shape[2]
        depth = volumes.shape[3]

        filename = os.path.join(self.output_dir, name + '.tfrecords')
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(volumes.shape[0]):
            volume_raw = volumes[index].ravel()
            label_raw = labels[index].tostring()
            example = tf.train.Example(
                features=self.features_encode(label_raw, volume_raw, rows, cols, depth)
            )
            writer.write(example.SerializeToString())
        writer.close()

class TwoPathwayProcessTup(AbstractProcessTup):

    def features_encode(self, label_raw, volume_raw, volume_raw_ss, rows, cols, depth, rows_ss, cols_ss, depth_ss):
        features = tf.train.Features(
            feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'height_ss': _int64_feature(rows_ss),
                'width_ss': _int64_feature(cols_ss),
                'depth_ss': _int64_feature(depth_ss),
                'volume_seg': _bytes_feature(label_raw),
                'volume_raw': _floats_feature(volume_raw),
                'volume_raw_ss': _floats_feature(volume_raw_ss)
            }
        )
        return features

    def features_decode(self, serialized_example):
        example = tf.parse_single_example(
            serialized_example,
            features={
                'volume_raw': tf.FixedLenFeature([np.prod(self.data_loader.segment_size)], tf.float32),
                'volume_raw_ss': tf.FixedLenFeature([np.prod(self.data_loader.segment_size_ss)], tf.float32),
                'volume_seg': tf.FixedLenFeature([], tf.string),
            }
        )
        volume = example['volume_raw']
        volume.set_shape([np.prod(self.data_loader.segment_size)])

        volume_ss = example['volume_raw_ss']
        volume_ss.set_shape([np.prod(self.data_loader.segment_size_ss)])

        label = tf.decode_raw(example['volume_seg'], tf.uint8)
        label.set_shape([np.prod(self.data_loader.segment_size-2*self.data_loader.crop_by)])
        return volume, label, volume_ss

    #def convert_to(self, volumes, labels, volumes_ss, name):
    def convert_to(self, *vals):
        volumes, labels, volumes_ss, name = vals
        rows = volumes.shape[1]
        cols = volumes.shape[2]
        depth = volumes.shape[3]
        rows_ss = volumes_ss.shape[1]
        cols_ss = volumes_ss.shape[2]
        depth_ss = volumes_ss.shape[3]

        filename = os.path.join(self.output_dir, name + '.tfrecords')
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(volumes.shape[0]):
            volume_raw = volumes[index].ravel()
            volume_raw_ss = volumes_ss[index].ravel()
            label_raw = labels[index].tostring()
            example = tf.train.Example(
                features=self.features_encode(label_raw, volume_raw, volume_raw_ss, rows, cols, depth, rows_ss, cols_ss, depth_ss)
            )
            writer.write(example.SerializeToString())
        writer.close()


class DistMapProcessTup(AbstractProcessTup):

    def features_encode(self, label_raw, volume_raw, distmap_raw, rows, cols, depth):
        features = tf.train.Features(
            feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'volume_seg': _bytes_feature(label_raw),
                'volume_raw': _floats_feature(volume_raw),
                'distmap_raw': _bytes_feature(distmap_raw)
            }
        )
        return features

    def features_decode(self, serialized_example):
        example = tf.parse_single_example(
            serialized_example,
            features={
                'volume_raw': tf.FixedLenFeature([np.prod(self.data_loader.segment_size)], tf.float32),
                'distmap_raw': tf.FixedLenFeature([], tf.string),
                'volume_seg': tf.FixedLenFeature([], tf.string),
            }
        )
        volume = example['volume_raw']
        volume.set_shape([np.prod(self.data_loader.segment_size)])

        distmap = tf.decode_raw(example['distmap_raw'], tf.uint8)
        distmap.set_shape([np.prod(self.data_loader.segment_size-2*self.data_loader.crop_by)])

        label = tf.decode_raw(example['volume_seg'], tf.uint8)
        label.set_shape([np.prod(self.data_loader.segment_size-2*self.data_loader.crop_by)])
        return volume, label, distmap

    def convert_to(self, *vals):
        volumes, labels, distmap, name = vals
        rows = volumes.shape[1]
        cols = volumes.shape[2]
        depth = volumes.shape[3]

        filename = os.path.join(self.output_dir, name + '.tfrecords')
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(volumes.shape[0]):
            volume_raw = volumes[index].ravel()
            distmap_raw = distmap[index].ravel()
            label_raw = labels[index].tostring()
            example = tf.train.Example(
                features=self.features_encode(label_raw, volume_raw, distmap_raw, rows, cols, depth)
            )
            writer.write(example.SerializeToString())
        writer.close()

class DenoiseProcessTup(AbstractProcessTup):

    def features_encode(self, label_raw, volume_raw, distmap_raw, rows, cols, depth):
        features = tf.train.Features(
            feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'volume_seg': _bytes_feature(label_raw),
                'volume_raw': _floats_feature(volume_raw),
                'denoise_raw': _floats_feature(distmap_raw)
            }
        )
        return features

    def features_decode(self, serialized_example):
        example = tf.parse_single_example(
            serialized_example,
            features={
                'volume_raw': tf.FixedLenFeature([np.prod(self.data_loader.segment_size)], tf.float32),
                'denoise_raw': tf.FixedLenFeature([np.prod(self.data_loader.segment_size)], tf.float32),
                'volume_seg': tf.FixedLenFeature([], tf.string),
            }
        )
        volume = example['volume_raw']
        volume.set_shape([np.prod(self.data_loader.segment_size)])

        denoise = example['denoise_raw']
        denoise.set_shape([np.prod(self.data_loader.segment_size)])

        label = tf.decode_raw(example['volume_seg'], tf.uint8)
        label.set_shape([np.prod(self.data_loader.segment_size-2*self.data_loader.crop_by)])
        return volume, label, denoise 

    def convert_to(self, *vals):
        volumes, labels, denoise, name = vals
        rows = volumes.shape[1]
        cols = volumes.shape[2]
        depth = volumes.shape[3]

        filename = os.path.join(self.output_dir, name + '.tfrecords')
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(volumes.shape[0]):
            volume_raw = volumes[index].ravel()
            denoise_raw = denoise[index].ravel()
            label_raw = labels[index].tostring()
            example = tf.train.Example(
                features=self.features_encode(label_raw, volume_raw, denoise_raw, rows, cols, depth)
            )
            writer.write(example.SerializeToString())
        writer.close()

class RecordWriter(object):
    def __init__(self, data_loader, ProcessTupClass, num_of_threads=4):
        self.data_loader = data_loader
        self.ProcessTupClass = ProcessTupClass
        self.num_of_threads = num_of_threads

    def write_records(self, output_dir):
        with Pool(self.num_of_threads) as p:
            pmap_results = list(
                zip(
                    *p.map(self.ProcessTupClass(self.data_loader, 'train', output_dir),
                           self.data_loader.train_tups)
                )
            )
            no_train_examples_dict = dict(pmap_results[0])
            no_train_class_dict = dict(pmap_results[1])
        no_validation_examples_dict = {}

        if self.data_loader.validation_tups:
            with Pool(self.num_of_threads) as p:
                pmap_results = list(
                    zip(
                        *p.map(
                            self.ProcessTupClass(self.data_loader, 'validation', output_dir),
                            self.data_loader.validation_tups)
                    )
                )
                no_validation_examples_dict = dict(pmap_results[0])
        data = {'train_examples':no_train_examples_dict,
                'train_classes':no_train_class_dict,
                'validation_examples':no_validation_examples_dict,
                'stride':self.data_loader.stride.tolist(),
                'segment_size_in':self.data_loader.segment_size.tolist(),
                'segment_size_out':(self.data_loader.segment_size-self.data_loader.crop_by).tolist(),
                'train_tups':self.data_loader.train_tups,
                'validation_tups':self.data_loader.validation_tups,
                'test_tups':self.data_loader.test_tups
               }
        with open(os.path.join(output_dir, 'meta_data.txt'), 'w') as outfile:
            json.dump(dict(data), outfile)

class RecordReader(object):
    def __init__(self, process_tup):
        self.ptc = process_tup

    def read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        return self.ptc.features_decode(serialized_example)

    def input_pipeline(self, train, batch_size, num_epochs, record_dir):
        read_threads = 20
        if not num_epochs:
            num_epochs = None
        search_string = os.path.join(record_dir,
                                     'train' if train else 'validation')
        search_string += '*'
        filenames = glob.glob(search_string)
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=True)
        example_list = [self.read_and_decode(filename_queue)
                        for _ in range(read_threads)]
        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * batch_size
        return tf.train.shuffle_batch_join(
            example_list, batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue, allow_smaller_final_batch=True)

    def get_weighting(self, filename):
        with open(filename, 'r') as f:
            meta_dict = json.load(f)
        train_classes = meta_dict['train_classes']
        weighting = np.sum(np.vstack([x for _, x in train_classes.items()]), axis = 0 )
        weighting = weighting/np.sum(weighting)
        weighting = 1 - weighting
        print(np.sum(weighting))
        weighting = weighting/np.sum(weighting)
        return weighting




