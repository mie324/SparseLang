import tensorflow as tf
import sys
import numpy as np
from pprint import pprint


def print_config(config):
    print("##################   Current Config   ################")
    pprint({attr: config.__getattribute__(attr) for attr in dir(config) if not attr.startswith('__')})


def print_flags(FLAGS):
    print("##################   Current Flags   ################")
    for key, value in tf.app.flags.FLAGS.flag_values_dict().items():
        print("{}:  {}".format(key, value))
        sys.stdout.flush()


def print_variable():
    all_var = tf.trainable_variables()
    print("##############################\n")
    lstm_var = []
    embedding_var = []
    output_var = []

    for var in all_var:
        if "sparse" in var.name:
            lstm_var.append(var)
        elif "embedding" in var.name:
            embedding_var.append(var)
        elif "softmax" in var.name:
            output_var.append(var)

    def print_var(var_list, name):
        total_num = 0
        for var in var_list:
            num = np.product([xi.value for xi in var.get_shape()])
            total_num += num
            print(var.name, num)
        print("#######      Total   {}: {}       ########".format(name, total_num))

    print_var(lstm_var,name="lstm")
    print_var(embedding_var,name="embedding")
    print_var(output_var,name="output")

    print("\n##############################")


def get_num_para():
    return np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()])
