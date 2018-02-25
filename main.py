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
from oxcnn.cnn import CNN

FLAGS = None

def train(FLAGS):
    cnn = CNN()
    module = importlib.import_module(FLAGS.model) 
    cnn.train(
        FLAGS.tfr_dir, 
        FLAGS.save_dir, 
        module,
        FLAGS.num_epochs,
        FLAGS.batch_size,
        FLAGS.num_save_every,
        FLAGS.model_file
    )

def test(FLAGS):
    cnn = CNN()
    module = importlib.import_module(FLAGS.model) 
    cnn.test(
        FLAGS.save_dir, 
        cnn.get_testdata(FLAGS.test_data_file), 
        FLAGS.model_file, 
        module,
        FLAGS.batch_size
    )

def write(FLAGS):
    cnn = CNN()
    module = importlib.import_module(FLAGS.model) 
    cnn.write_records(
        FLAGS.save_dir, 
        FLAGS.data_dir, 
        FLAGS.dir_type, 
        module
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        required=True,
        type=str,
        action='store')
    subparsers = parser.add_subparsers(help='sub-command help')
    parser_train = subparsers.add_parser('train', help='train help')
    parser_train.add_argument('--tfr_dir', type=str, required=True)
    parser_train.add_argument('--save_dir', type=str, required=True)
    parser_train.add_argument('--num_epochs', type=int, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--test_data', type=str, required=False)
    parser_train.add_argument('--num_save_every', type=int, required=False)
    parser_train.add_argument('--num_batches_val', type=int, required=False)
    parser_train.add_argument('--model_file', type=str, required=False)
    parser_train.set_defaults(func=train)

    parser_test = subparsers.add_parser('test', help='test help')
    parser_test.add_argument('--save_dir', type=str, required=True)
    parser_test.add_argument('--test_data_file', type=str, required=True)
    parser_test.add_argument('--model_file', type=str, required=True)
    parser_test.add_argument('--batch_size', type=int, required=True)
    parser_test.set_defaults(func=test)

    parser_write = subparsers.add_parser('write', help='write help')
    parser_write.add_argument('--save_dir', type=str, required=True)
    parser_write.add_argument('--data_dir', type=str, required=True)
    parser_write.add_argument('--dir_type', type=str, required=False)
    parser_write.set_defaults(func=write)

    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS, [sys.argv[0]] + unparsed)
    FLAGS.func(FLAGS)
    #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
