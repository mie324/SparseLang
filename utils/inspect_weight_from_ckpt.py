import os
import csv
import tensorflow as tf
from tensorflow.contrib.framework.python.framework import checkpoint_utils
import numpy as np
from utils.visualization_utils import get_heat_map, get_histogram
from utils.misc import pprint
from matplotlib.mlab import PCA

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("ckpt_path", None, "The path to ckpt")
tf.flags.DEFINE_string("output_path", None, "The path to store csv file")
tf.flags.DEFINE_boolean("heat_map", False, "whether output heat map")
tf.flags.DEFINE_boolean("histogram", False, "whether output histogram")
tf.flags.DEFINE_boolean("pca", False, "whether or not perform PCA Analysis")
tf.flags.DEFINE_string("ignore", None, "what kind of params to ignore")
tf.logging.set_verbosity(tf.logging.INFO)


def save_weight_csv(output_dir, var_dic):
    with open(output_dir, 'w') as csvfile:
        fieldnames = ['weight']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(var_dic)


def get_stat_single(key, value):
    print("\nVariable Name: {}".format(key))
    print("Variable shape: {}".format(value.shape))
    num_para = np.prod(value.shape)

    mean = np.mean(value)
    variance = np.var(value)
    print("Mean: {}".format(mean))
    print("Variance: {}".format(np.var(variance)))

    normalized_value = (value - mean) / np.sqrt(variance + 1e-8)
    threshhold = 0.05

    trivial_value = np.where(abs(normalized_value) < threshhold, 1, 0)
    print("Trivial Value: {}, {}".format(np.sum(trivial_value), np.sum(trivial_value) / num_para))
    round_value = np.where(abs(normalized_value) < threshhold, 0, normalized_value)

    if FLAGS.heat_map:
        # get_grey_image(key, value, "_original")
        # get_grey_image(key, normalized_value, "_normalized")
        # get_grey_image(key, round_value, "_round")
        get_grey_image(key, np.abs(value), "_heat_abs")
        get_grey_image(key, np.abs(round_value), "_heat_roundabs")

    if FLAGS.histogram:
        # get_hist(key, value, "_original")
        # get_hist(key, normalized_value, "_normalized")
        # get_hist(key, round_value, "_round")
        get_hist(key, np.abs(value), "_abs")
        get_hist(key, np.abs(round_value), "_roundabs")

    if FLAGS.pca:
        PCA_analysis(key, value)


def PCA_analysis(key, value):
    if len(value.shape) == 1:
        return
    P, D, Q = np.linalg.svd(value, full_matrices=False)
    print(D.shape)
    sorted_D = -np.sort(-D)
    print(sorted_D)
    get_hist(key, D, "_singular_value")
    # X_a = np.matmul(np.matmul(P, np.diag(D)), Q)


def get_stat(var_dict):
    for key in var_dict.keys():
        get_stat_single(key.replace("/", "_"), var_dict[key])
    print("Finished!")


def get_hist(key, value, others=""):
    out_file = os.path.join(FLAGS.output_path, key + others)
    print(out_file)
    get_histogram(value, out_file, title=key + others, bins=45)


def get_grey_image(key, value, others=""):
    if len(value.shape) == 1:
        return
    out_file = os.path.join(FLAGS.output_path, key + others)
    print(out_file)
    get_heat_map(key, np.array(value), out_file)


def main(unarg):
    if not FLAGS.ckpt_path:
        raise ValueError("ckpt_path must be specified")
    if not tf.gfile.Exists(FLAGS.ckpt_path):
        raise ValueError("The path does not exist!")
    if not tf.gfile.Exists(FLAGS.output_path):
        print("Create output file!")
        tf.gfile.MkDir(FLAGS.output_path)

    # var_dict = load_weight(FLAGS.ckpt_path)
    # get_stat(var_dict)

    var_name, var_dict = inspect_srnn_ckpt(FLAGS.ckpt_path)
    get_stat(var_dict)


def inspect_srnn_ckpt(checkpoint_dir):
    vars = checkpoint_utils.list_variables(checkpoint_dir)
    forbidden_words = ["Adam", "beta", "embedding", "step", "Variable", "Bias"]
    var_name = []
    for var in vars:
        if any([x in var[0] for x in forbidden_words]):
            continue
        else:
            var_name.append(var[0])
    pprint(var_name)

    var_dict = {}

    for name in var_name:
        var_dict[name] = checkpoint_utils.load_variable(checkpoint_dir, name)

    return var_name, var_dict


if __name__ == '__main__':
    tf.app.run()
    # checkpoint_dir = "/scratch/gobi1/zycluke/SparseLang/small_model-28834"
    # var_dict = inspect_srnn_ckpt(checkpoint_dir)
    # pprint(var_dict)
    # python inspect_weight_from_ckpt.py --ckpt_path "/scratch/gobi1/zycluke/SparseLang/small_model-28834" --output_path temp_visual
