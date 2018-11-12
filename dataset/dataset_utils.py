'''
Dataset Utils: Complete Sentence and Bucket Iterator
'''

import tensorflow as tf
import os
import sys
import numpy as np
from dataset.reader import _build_vocab

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", 32, "Batch size.")
tf.flags.DEFINE_integer("num_buckets", 5, "Put data into similar-length buckets.")


def train_input_fn(filename):
    dataset = split_dataset(filename)
    dataset = word_to_id(filename, dataset)
    dataset = tf.data.Dataset.from_tensor_slices(dataset)

    # dataset = dataset.map(map_func=parse_exmp)
    # dataset = bucket_batch(dataset, batch_size=FLAGS.batchsize)
    print(dataset)
    return dataset


def split_dataset(filename):
    with tf.gfile.GFile(filename, "r") as f:
        sentences = f.read().split('\n')
        for i, sentence in enumerate(sentences):
            sentences[i] = sentence.split()

        return np.asarray(sentences)


def bucket_batch(dataset, batch_size):
    def batching_func(x):
        return x.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None, 1]),  # src
                                                         tf.TensorShape([None, 1]),  # tgt
                                                         tf.TensorShape([])),  # tgt_len
                              padding_values=(0,  # src
                                              0,  # tgt
                                              0))

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


def word_to_id(filename, dataset):
    word_to_id = _build_vocab(filename)
    # sentence_length = []
    for i, sentence in enumerate(dataset):
        dataset[i] = [word_to_id[word] for word in sentence if word in word_to_id]
        # dataset[i] = tf.convert_to_tensor(dataset[i])
        # sentence_length.append(len(dataset[i]))
    return dataset


def dataset_generator(filename):
    dataset = split_dataset(filename)
    dataset = word_to_id(filename, dataset)
    print('hhah')
    # input =
    # output =

    bucket_dataset = bucket_batch(dataset, batch_size)

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))


def dataset_textline(filename, word_to_id):
    dataset = tf.data.TextLineDataset(filename)
    dataset = dataset.map(map_func=lambda string: tf.string_split([string]).values)
    dataset = dataset.map(map_func=lambda sentence: parse_exmp(sentence, word_to_id))

    return dataset

# https://cs230-stanford.github.io/tensorflow-input-data.html: Creating the vocabulary

def parse_exmp(sentence, word_to_id):
    sentence_token = sentence.split()

    return sentence


def main(unused_argv):
    # Testing function
    print('Hello world')
    word_to_id = _build_vocab("../data/ptb.test.txt")
    exmp = dataset_textline("../data/ptb.test.txt", word_to_id)
    iterator = exmp.make_one_shot_iterator()
    input = iterator.get_next()

    print('Start_session!')

    with tf.Session() as sess:
        # sess.run(iterator.initializer)

        for i in range(10):
            try:
                # cur_input = sess.run(input)
                # cur_label = sess.run(label)
                # cur_seq = sess.run(seq_len)
                # print("input",cur_input)
                # print("label",cur_label)
                # print("seq_len",cur_seq)
                cur_input = sess.run(input)
                print(cur_input)

            except tf.errors.OutOfRangeError:
                print("End of dataset")
                print('current ', i)
                break
                # sess.run(iterator.initializer)

    print('finished!')


if __name__ == '__main__':
    tf.app.run(main)
