'''
Dataset Utils: Complete Sentence and Bucket Iterator
'''

import tensorflow as tf
import os
import numpy as np
from reader import _build_vocab as _build_vocab

def convert_to_sentence():
    pass


def split_dataset(filename):
    with tf.gfile.GFile(filename, "r") as f:
        # sentences = f.read().decode("utf-8").split('\n')
        sentences = f.read().split('\n')
        for i,sentence in enumerate(sentences):
            sentences[i] = sentence.split()


        return np.asarray(sentences)


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

def word_to_id(filename,dataset):
    word_to_id = _build_vocab(filename)
    for i,sentence in enumerate(dataset):
        dataset[i] = [word_to_id[word] for word in sentence if word in word_to_id]
    return dataset

def dataset_generator(filename):
    dataset = split_dataset(filename)
    dataset = word_to_id(filename,dataset)
    print('hhah')
    # input =
    # output =


    bucket_dataset = bucket_batch(dataset, batch_size)


    dataset = tf.data.Dataset.from_tensor_slices((features, labels))



def train_input(filename):
    dataset = dataset_generator(filename)
    dataset = tf.Dataset.from_numpy

if __name__ == '__main__':
    print(os.getcwd())
    filename = '../data/ptb.train.txt'
    # splited = split_dataset(filename)
    dataset_generator(filename)
