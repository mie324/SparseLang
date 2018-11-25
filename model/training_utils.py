import tensorflow as tf
import numpy as np
import sys

FLAGS = tf.flags.FLAGS

# Optimization
tf.flags.DEFINE_string("optimizer", "adam", "sgd | adam")
tf.flags.DEFINE_float("max_global_gradient_norm", 5000.0, "Clip gradients to this norm.")
tf.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.flags.DEFINE_boolean("clip_by_global", True, "Whether use clip by global or by norm")
tf.flags.DEFINE_boolean("colocate_gradients_with_ops", True, "Whether try colocating gradients with corresponding op")
# Learning rate
tf.flags.DEFINE_float("learning_rate", 0.0002, "Learning rate. Adam: 0.001 | 0.0001")
tf.flags.DEFINE_integer("warmup_steps", 1000, "How many steps we inverse-decay learning.")
tf.flags.DEFINE_string("warmup_scheme", "t2t",
                       "How to warmup learning rates. Options include:t2t: Tensor2Tensor's way, start with lr 100 times smaller, then exponentiate until the specified lr.")
tf.flags.DEFINE_string("decay_scheme", "luke", """How we decay learning rate. Options include: luong234: after 2/3 num train steps, we start halving the learning rate for 4 times before finishing.\
   luong5: after 1/2 num train steps, we start halving the learning rate for 5 times before finishing.\
   luong10: after 1/2 num train steps, we start halving the learning rate for 10 times before finishing.""")
tf.flags.DEFINE_boolean("summarize_gradients", False, "summarize_gradients")
tf.flags.DEFINE_boolean("summarize_weights", False, "summarize_weights")
tf.flags.DEFINE_float("adam_epsilon", 1e-4, "adam_epsilon")


def setup_learning_rate(FLAGS, global_step):
    def _get_learning_rate_warmup(FLAGS, global_step, learning_rate):
        """Get learning rate warmup."""
        warmup_steps = FLAGS.warmup_steps
        warmup_scheme = FLAGS.warmup_scheme
        tf.logging.info("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" % (
            FLAGS.learning_rate, warmup_steps, warmup_scheme))

        # Apply inverse decay if global steps less than warmup steps.
        # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
        # When step < warmup_steps,
        #   learing_rate *= warmup_factor ** (warmup_steps - step)
        if warmup_scheme == "t2t":
            # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
            warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
            inv_decay = warmup_factor ** (tf.to_float(warmup_steps - global_step))
        else:
            raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

        return tf.cond(global_step < FLAGS.warmup_steps, lambda: inv_decay * learning_rate,
                       lambda: learning_rate, name="learning_rate_warump_cond")

    def _get_learning_rate_decay(FLAGS, global_step, learning_rate):
        """Get learning rate decay."""
        if FLAGS.decay_scheme in ["luong5", "luong10", "luong234", "luke"]:
            decay_factor = 0.5
            if FLAGS.decay_scheme == "luong5":
                start_decay_step = int(FLAGS.num_train_steps / 2)
                decay_times = 5
            elif FLAGS.decay_scheme == "luong10":
                start_decay_step = int(FLAGS.num_train_steps / 2)
                decay_times = 10
            elif FLAGS.decay_scheme == "luong234":
                start_decay_step = int(FLAGS.num_train_steps * 2 / 3)
                decay_times = 4
            elif FLAGS.decay_scheme == "luke":
                start_decay_step = int(FLAGS.num_train_steps / 3)
                decay_times = 4
            remain_steps = FLAGS.num_train_steps - start_decay_step
            decay_steps = int(remain_steps / decay_times)
        elif not FLAGS.decay_scheme:  # no decay
            start_decay_step = FLAGS.num_train_steps
            decay_steps = 0
            decay_factor = 1.0
        elif FLAGS.decay_scheme:
            raise ValueError("Unknown decay scheme %s" % FLAGS.decay_scheme)

        tf.logging.info("decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
                        "decay_factor %g" % (FLAGS.decay_scheme, start_decay_step, decay_steps, decay_factor))

        return tf.cond(global_step < start_decay_step, lambda: learning_rate,
                       lambda: tf.train.exponential_decay(learning_rate, (global_step - start_decay_step),
                                                          decay_steps, decay_factor, staircase=True),
                       name="learning_rate_decay_cond")

    learning_rate = tf.constant(FLAGS.learning_rate)
    # warm-up
    learning_rate = _get_learning_rate_warmup(FLAGS, global_step, learning_rate)
    # decay
    learning_rate = _get_learning_rate_decay(FLAGS, global_step, learning_rate)

    return learning_rate


def get_train_op_and_metrics(FLAGS, loss):
    global_step = tf.train.get_or_create_global_step()
    learning_rate = setup_learning_rate(FLAGS, global_step)

    # Optimizer
    if FLAGS.optimizer == "sgd":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif FLAGS.optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=FLAGS.adam_epsilon)
    elif FLAGS.optimizer == "adadelta":
        optimizer = tf.train.AdadeltaOptimizer(learning_rate, epsilon=FLAGS.adam_epsilon)
    elif FLAGS.optimizer == "adagrad":
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif FLAGS.optimizer == "momentum":
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif FLAGS.optimizer == "ftrl":
        optimizer = tf.train.FtrlOptimizer(learning_rate)
    elif FLAGS.optimizer == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    elif FLAGS.optimizer == "proximalgd":
        optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate)
    elif FLAGS.optimizer == "proximaladagrad":
        optimizer = tf.train.ProximalAdagradOptimizer(learning_rate)
    else:
        raise ValueError("Unknown Optimizer {}".format(FLAGS.optimizer))

    train_op = tf.contrib.training.create_train_op(
        loss,
        optimizer,
        global_step=global_step,
        update_ops=None,
        variables_to_train=None,
        transform_grads_fn=None,
        summarize_gradients=FLAGS.summarize_gradients,
        aggregation_method=None,
        colocate_gradients_with_ops=FLAGS.colocate_gradients_with_ops,
        check_numerics=True)

    if FLAGS.summarize_weights:
        print("Summarize weights")
        variables_to_train = tf.trainable_variables()
        add_weight_summary(variables_to_train)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(train_op, update_ops)

    train_metrics = {"learning_rate": learning_rate}

    return train_op, train_metrics


def record_scalars(metric_dict):
    print("{} metrics to write".format(len(metric_dict)))
    sys.stdout.flush()
    for key, value in metric_dict.items():
        print("Record {} to summary".format(key))
        sys.stdout.flush()
        tf.summary.scalar(name=key, tensor=value)


def add_weight_summary(variable):
    """Add summaries to variable norm.

    Args:
      variable: trainable variable

    Returns:
      The list of created summaries.
    """
    summaries = []
    for var in variable:
        summaries.append(
            tf.summary.histogram(var.name + '_weight_norm', tf.norm(var)))
        summaries.append(
            tf.summary.scalar(var.name + '_weight_norm', tf.norm(var)))

    return summaries

def get_num_para():
    return np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.all_variables()])
