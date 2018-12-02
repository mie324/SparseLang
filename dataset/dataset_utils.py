'''
Dataset Utils: Complete Sentence and Bucket Iterator
'''

import tensorflow as tf
import os
import numpy as np
from dataset.reader import _build_vocab
import argparse

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", 128, "Batch size.")
tf.flags.DEFINE_integer("buffer_size", 30, "Buffer size.")


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

def generate_labels(filename , path_labels):
    label_file = open(path_labels, "a+")
    with tf.gfile.GFile(filename, "r") as f:
        sentences = f.read().split('\n')
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) < 2:
                sentence = sentence
            else:
                sentence = sentence.split(None, 1)[1] + ' <unk> '

            label_file.write(sentence + "\r\n")
    label_file.close


def load_dataset_from_text(path_txt, vocab):
    """Create tf.data Instance from txt file
    Args:
        path_txt: (string) path containing one example per line
        vocab: (tf.lookuptable)
    Returns:
        dataset: (tf.Dataset) yielding list of ids of tokens for each example
    """
    # Load txt file, one example per line
    dataset = tf.data.TextLineDataset(path_txt)

    # Convert line into list of tokens, splitting by white space
    dataset = dataset.map(lambda string: tf.string_split([string]).values)

    # Lookup tokens to return their ids
    dataset = dataset.map(lambda tokens: (vocab.lookup(tokens), tf.size(tokens)))

    return dataset


def input_fn(mode, sentences, labels, params):
    """Input function for NER
    Args:
        mode: (string) 'train', 'eval' or any other mode you can think of
                     At training, we shuffle the data and have multiple epochs
        sentences: (tf.Dataset) yielding list of ids of words
        datasets: (tf.Dataset) yielding list of ids of tags
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    # Load all the dataset in memory for shuffling is training
    is_training = (mode == 'train')
    buffer_size = params.buffer_size if is_training else 1

    # Zip the sentence and the labels together
    dataset = tf.data.Dataset.zip((sentences, labels))

    # Create batches and pad the sentences of different length
    padded_shapes = ((tf.TensorShape([None]),  # sentence of unknown size
                      tf.TensorShape([])),     # size(words)
                     (tf.TensorShape([None]),  # labels of unknown size
                      tf.TensorShape([])))     # size(tags)

    padding_values = ((params.id_pad_word,   # sentence padded on the right with id_pad_word
                       0),                   # size(words) -- unused
                      (params.id_pad_tag,    # labels padded on the right with id_pad_tag
                       0))                   # size(tags) -- unused


    dataset = (dataset
        .shuffle(buffer_size=buffer_size)
        .padded_batch(params.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
        .prefetch(1)  # make sure always have one batch ready to serve
    )

    # Create initializable iterator from this dataset so that we can reset at each epoch
    iterator = dataset.make_initializable_iterator()

    # Query the output of the iterator for input to the model
    ((sentence, sentence_lengths), (labels, _)) = iterator.get_next()
    init_op = iterator.initializer

    # Build and return a dictionnary containing the nodes / ops
    inputs = {
        'sentence': sentence,
        'labels': labels,
        'sentence_lengths': sentence_lengths,
        'iterator_init_op': init_op
    }

    return inputs

def main(args):
    print(os.getcwd())
    filename = '../data/ptb.train.txt'
    path_words = "./words.txt"

    # split_dataset(filename)
    # outFile_name = "./words.txt"
    # saveTokens(outFile_name, filename)
    # words = tf.contrib.lookup.index_table_from_file("./words.txt", num_oov_buckets=1)
    # dataset = split_dataset(filename)
    # dataset = word_to_id(filename, dataset)

    # Load Vocabularies
    words = tf.contrib.lookup.index_table_from_file(path_words, num_oov_buckets=1) # num_oov_buckets specifies the number of buckets created for unknow words， set to 1

    path_train_sentences = '../data/ptb.train.txt'
    path_train_labels = './train.labels.txt'
    path_valid_sentences = '../data/ptb.valid.txt'
    path_valid_labels = './valid.labels.txt'
    path_test_sentences = '../data/ptb.test.txt'
    path_test_labels = './test.labels.txt'
    generate_labels(path_train_sentences,path_train_labels)
    generate_labels(path_valid_sentences, path_valid_labels)
    generate_labels(path_test_sentences, path_test_labels)

    train_sentences = load_dataset_from_text(path_train_sentences, words)
    train_labels = load_dataset_from_text(path_train_labels, words)
    eval_sentences = load_dataset_from_text(path_valid_sentences, words)
    eval_labels = load_dataset_from_text(path_valid_labels, words)
    test_sentences = load_dataset_from_text(path_test_sentences, words)
    test_labels = load_dataset_from_text(path_test_labels, words)

    # Specify other parameters for the dataset and the model
    # args.eval_size = args.dev_size
    # args.buffer_size = args.train_size  # buffer size for shuffling
    args.id_pad_word = words.lookup(tf.constant(args.pad_word))
    args.id_pad_tag = words.lookup(tf.constant(args.pad_tag))

    # Create the two iterators over the two datasets
    train_inputs = input_fn('train', train_sentences, train_labels, args)
    eval_inputs = input_fn('eval', eval_sentences, eval_labels, args)
    test_inputs = input_fn('eval', test_sentences, test_labels, args)

    print("- done.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_size', default=int)
    parser.add_argument('--buffer_size', default=30, help="buffer size for shuffling")
    parser.add_argument('--id_pad_word', default=int)
    parser.add_argument('--id_pad_tag', default=int)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--pad_word', default='<unk>')
    parser.add_argument('--pad_tag', default='<unk>')
    args = parser.parse_args()

    main(args)




