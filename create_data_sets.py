import os
import json
import numpy as np
from collections import namedtuple, Counter
import tensorflow as tf


tf.flags.DEFINE_string("train_img_dir", "data/img/train2014/", "Training image directory.")
tf.flags.DEFINE_string("val_img_dir", "data/img/val2014/", "Validation image directory.")
tf.flags.DEFINE_string("train_captions", "data/stair_captions_v1.1_train.json", "Training caption file.")
tf.flags.DEFINE_string("val_captions", "data/stair_captions_v1.1_val.json", "Validation caption file.")
tf.flags.DEFINE_string("out_dir", "data/tfrecords/", "Output TFRecords directiory.")
tf.flags.DEFINE_integer("min_word_count", 4, "The minimum number of occurrences of each word in th training set for includion in the vocab.")
tf.flags.DEFINE_string("word_list_file", "data/dictionary.txt", "Output word list file.")

FLAGS = tf.flags.FLAGS