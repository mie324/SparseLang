import time
import os
import shutil
import numpy as np
from model.model_config import *
from model.model import *
from utils.misc import print_config, print_flags, get_num_para, print_variable

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("data_path", 'data', "Where the training/test data is stored.")
tf.flags.DEFINE_string("save_path", None, "Model output directory.")
tf.flags.DEFINE_boolean("debug", False, "whether in debug mode or not.")
tf.flags.DEFINE_boolean("allow_growth", True, "GPU allow_growth.")
tf.flags.DEFINE_float("gpu_memory_fraction", 1, "gpu_memory_fraction.")

tf.flags.DEFINE_boolean("larger_hidden_size", False, "whether increase the number of hidden units given sparsity.")


def main(unused_argv):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")
    if FLAGS.debug:
        print("########## Debug Mode ##########")
        if os.path.exists(FLAGS.save_path):
            print("Remove previous model cache")
            shutil.rmtree(FLAGS.save_path)
    if not FLAGS.save_path:
        raise ValueError("save_path need to be specified")
    if not os.path.exists(FLAGS.save_path):
        print("Create save directory {}".format(FLAGS.save_path))
        os.makedirs(FLAGS.save_path)

    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, _ = raw_data

    config = get_config()
    if FLAGS.larger_hidden_size:
        config.hidden_size = int(config.hidden_size / np.sqrt(FLAGS.sparsity))
        print("Increase Hidden Size: {}".format(config.hidden_size))
    print_config(config)
    print_flags(FLAGS)

    eval_config = get_eval_config()

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, input_=train_input)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
            tf.summary.scalar("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = PTBModel(is_training=False, config=eval_config,
                                 input_=test_input)

        print("####### Total Number of Parameters: {}".format(get_num_para()))
        print_variable()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction,
                                    allow_growth=FLAGS.allow_growth)
        session_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session(config=session_config) as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                             verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, os.path.join(os.getcwd(), FLAGS.save_path) + "/model",
                              global_step=sv.global_step)


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


if __name__ == "__main__":
    tf.app.run()
    # python ptb_word_knet.py --data_path data --model large --useKnetOutput --useKnet
    # python main.py --data_path data --model_size small --model_type baseline --optimizer adam
    # python main.py --data_path data --model_size small --model_type sparse --optimizer adam --debug --save_path temp
