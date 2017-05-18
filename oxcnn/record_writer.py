import os
import numpy as np
import tensorflow as tf
import json
from multiprocessing import Pool
import glob

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

class StandardProcessTup(object):
    def __init__(self, data_loader, prefix,output_dir):
        self.data_loader = data_loader
        self.prefix = prefix
        self.output_dir = output_dir

    def __call__(self, tup):
        name = os.path.basename(tup[0].split('.')[0])
        x, y = self.data_loader.get_batch(tup)
        self.convert_to(x,y,self.prefix + '_' + name)
        return (name, len(x))

    def convert_to(self,volumes, labels, name):
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
                features=tf.train.Features(
                    feature={
                        'height': _int64_feature(rows),
                        'width': _int64_feature(cols),
                        'depth': _int64_feature(depth),
                        'volume_seg': _bytes_feature(label_raw),
                        'volume_raw': _floats_feature(volume_raw)
                    }
                )
            )
            
            writer.write(example.SerializeToString())
        writer.close()

class RecordWriter(object):
    def __init__(self,data_loader,ProcessTupClass):
        self.data_loader = data_loader
        self.ProcessTupClass = ProcessTupClass

    def write_records(self,output_dir):

        with Pool(2) as p:
            no_train_examples_dict = dict( p.map(self.ProcessTupClass(self.data_loader,'train',output_dir),self.data_loader.train_tups))

        no_validation_examples_dict = {}

        with Pool(2) as p:
            no_validation_examples_dict = dict( p.map(self.ProcessTupClass(self.data_loader,'validation',output_dir), self.data_loader.validation_tups))

        data={'train_examples':no_train_examples_dict,
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
    def __init__(self,segment_size,segment_size_out):
        self.segment_size = segment_size
        self.segment_size_out = segment_size_out

    def read_and_decode(self,filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        example = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'volume_raw': tf.FixedLenFeature([np.prod(self.segment_size)], tf.float32),
                'volume_seg': tf.FixedLenFeature([], tf.string),
            })
        #volume = tf.decode_raw(example['volume_raw'], tf.float32)
        volume = example['volume_raw']
        volume.set_shape([np.prod(self.segment_size)])

        label = tf.decode_raw(example['volume_seg'], tf.uint8)
        label.set_shape([np.prod(self.segment_size_out)])
        return volume, label

    def input_pipeline(self,train, batch_size, num_epochs, record_dir):
        read_threads = 20 
        if not num_epochs: num_epochs = None
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
        example_batch, label_batch = tf.train.shuffle_batch_join(
            example_list, batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue, allow_smaller_final_batch=True)
        return example_batch, label_batch
