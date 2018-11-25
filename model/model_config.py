"""Example / benchmark for building a PTB LSTM model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
To run:
$ python ptb_word_lm.py --data_path=simple-examples/data/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.flags.FLAGS


class Max_full_Config(object):
    """Small Sparse config."""
    init_scale = 0.04
    learning_rate = 0.5
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 2900
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class Same_size_Config(object):
    """Small Sparse config."""
    init_scale = 0.04
    learning_rate = 0.5
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 2500
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class Large_hidden_Config(object):
    """Small Sparse config."""
    init_scale = 0.04
    learning_rate = 0.5
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 10
    keep_prob = 1.
    lr_decay = 0.5
    batch_size = 10
    vocab_size = 10000


class SmallSparseConfig(object):
    """Small Sparse config."""
    init_scale = 0.04
    learning_rate = 0.5
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 400
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class DebugConfig(object):
    """Small Sparse config."""
    init_scale = 0.04
    learning_rate = 0.05
    max_grad_norm = 5
    num_layers = 1
    num_steps = 20
    hidden_size = 400
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class MediumSparseConfig(object):
    """Medium Sparse config."""
    init_scale = 0.1
    learning_rate = 0.05
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 700
    max_epoch = 16
    max_max_epoch = 50
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 25
    vocab_size = 10000


class LargeSparseConfig(object):
    """Large Sparse config."""
    init_scale = 0.1
    learning_rate = 0.05
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 250000
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 25
    vocab_size = 10000


class LargeConfig(object):
    """Large Knet config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 250000
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 25
    vocab_size = 10000


class TestSparseConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 0.05
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def get_config():
    if FLAGS.model_size == "small":
        return SmallSparseConfig()
    elif FLAGS.model_size == "medium":
        return MediumSparseConfig()
    elif FLAGS.model_size == "large":
        return LargeSparseConfig()
    elif FLAGS.model_size == "test":
        return TestSparseConfig()
    elif FLAGS.model_size == "debug":
        return DebugConfig()
    elif FLAGS.model_size == "customized":
        return Max_full_Config()
    elif FLAGS.model_size == "knet":
        return LargeConfig()
    elif FLAGS.model_size == "same_size":
        return Same_size_Config()
    elif FLAGS.model_size == "larger_hidden":
        return Large_hidden_Config()
    else:
        raise ValueError("Invalid model size: %s", FLAGS.model_size)


def get_eval_config():
    config = get_config()
    config.batch_size = 1
    config.num_steps = 1

    return config


def updata_config(config):
    raise NotImplementedError
