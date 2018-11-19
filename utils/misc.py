import tensorflow as tf
import sys
from pprint import pprint


def print_config(config):
    print("##################   Current Config   ################")
    pprint({attr: config.__getattribute__(attr) for attr in dir(config) if not attr.startswith('__')})


def print_flags(FLAGS):
    print("##################   Current Flags   ################")
    for key, value in tf.app.flags.FLAGS.flag_values_dict().items():
        print("{}:  {}".format(key,value))
        sys.stdout.flush()
