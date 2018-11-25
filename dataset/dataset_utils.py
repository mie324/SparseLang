'''
Dataset Utils: Complete Sentence and Bucket Iterator
'''

import tensorflow as tf
import os
import numpy as np
from dataset.reader import _build_vocab

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", 128, "Batch size.")

def convert_to_sentence():
    pass


def parse_exmp(source):
    # features = tf.parse_single_example(serial_exmp, features={'input': tf.VarLenFeature(tf.int32)})
    # source = tf.sparse_tensor_to_dense(features['input'], default_value=0)

    source = tf.cast(tf.reshape(source, (1, -1)), tf.int32)

    seq_len = tf.shape(source) + 1

    input = tf.concat(([0], source), 0)
    label = tf.concat((source, [0]), 0)

    return input, label, seq_len


def train_input_fn(filename):
    dataset = split_dataset(filename)
    dataset = word_to_id(filename, dataset)
    dataset = tf.data.Dataset.from_tensors(dataset)

    dataset = dataset.map(map_func=parse_exmp)
    dataset = bucket_batch(dataset, batch_size=FLAGS.batchsize)
    return dataset


def split_dataset(filename):
    with tf.gfile.GFile(filename, "r") as f:
        sentences = f.read().split('\n')
        for i, sentence in enumerate(sentences):
            sentences[i] = sentence.split()
        # return np.asarray(sentences)
        return sentences


def bucket_batch(dataset, batch_size):
    def batching_func(x):
        return x.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None, 1]),  # src
                                                         tf.TensorShape([None, 1]),  # tgt
                                                         tf.TensorShape([]),  # src_len
                                                         tf.TensorShape([])),  # tgt_len
                              padding_values=(0.0,  # src
                                              0.0,  # tgt
                                              0,  # src_len
                                              0)).prefetch(buffer_size=FLAGS.prefetch_buffer_size)

    if FLAGS.num_buckets > 1:

        def key_func(unused_1, unused_2, unused_3, unused_4, unused_5, unused_6, tgt_len):
            # Calculate bucket_width by maximum source sequence length.
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
            if FLAGS.tgt_max_len:
                bucket_width = (FLAGS.tgt_max_len + FLAGS.num_buckets - 1) // FLAGS.num_buckets
            else:
                bucket_width = 10

            # Bucket sentence pairs by the length of their source sentence and target
            # sentence.
            bucket_id = tgt_len // bucket_width
            return tf.to_int64(tf.minimum(FLAGS.num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        batched_dataset = dataset.apply(
            tf.contrib.data.group_by_window(key_func=key_func, reduce_func=reduce_func, window_size=FLAGS.batch_size))

    else:
        batched_dataset = batching_func(dataset)

    return batched_dataset

def saveTokens(outFile_name, filename):
    '''Function saves result of tokenFrequency into a text file.'''
    outFile = open(outFile_name, "a+")
    # outFile.write(uniqueLines)
    # outFile.close
    with tf.gfile.GFile(filename, "r") as f:
        sentences = f.read().split('\n')
        for i, sentence in enumerate(sentences):
            for word in sentence.split():
                outFile.write(word + "\r\n")
    outFile.close

def word_to_id(filename, dataset):
    word_to_id = _build_vocab(filename)
    # sentence_length = []
    label = []
    for i, sentence in enumerate(dataset):
        dataset[i] = [word_to_id[word] for word in sentence if word in word_to_id]
        label.append(dataset[i][1:] + [0])
        # sentence_length.append(len(dataset[i]))
    return dataset,label


def dataset_generator(filename):
    dataset = split_dataset(filename)
    dataset,label = word_to_id(filename, dataset)

    words = tf.contrib.lookup.index_table_from_file(dataset, num_oov_buckets=1)
    # features = dataset
    # labels = label
    features = np.asarray(dataset)
    labels = np.asarray(label)

    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))

    bucket_dataset = bucket_batch(dataset, batch_size)

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))


def train_input(filename):
    dataset = dataset_generator(filename)
    dataset = tf.Dataset.from_numpy

def main():
    print(os.getcwd())
    filename = '../data/ptb.train.txt'
    outFile_name = "./words.txt"
    split_dataset(filename)
    saveTokens(outFile_name, filename)
    words = tf.contrib.lookup.index_table_from_file("./words.txt", num_oov_buckets=1)
    dataset = split_dataset(filename)
    dataset = word_to_id(filename, dataset)
    print('haha')

    # dataset_generator(filename)
    # train_input_fn(filename)
if __name__ == '__main__':

    main()

