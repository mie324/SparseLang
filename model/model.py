import tensorflow as tf
from dataset import reader
import rnn_cell

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_bool("use_fp16", False, "Train using 16-bit floats instead of 32bit floats")
tf.flags.DEFINE_string("model_type", "baseline", "The type of rnn model| Default: baseline(normal_lstm)")
tf.flags.DEFINE_string("model_size", "small", "A type of model. Possible options are: small, medium, large.")
tf.flags.DEFINE_string("output_type", "linear", "The type of output layer | Default: linear | knet")


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config, input_):
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        emb_size = 600
        vocab_size = config.vocab_size

        # Slightly better results can be obtained with forget gate biases initialized to 1 but the hyperparameters of
        # the model would need to be different than reported in the paper.
        # Todo why is that?

        if FLAGS.model_type == "baseline":
            lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        elif FLAGS.model_type == "sparse":
            raise NotImplementedError
        elif FLAGS.model_type == "knet":
            lstm_cell = rnn_cell.BasicLSTMCell_knet(size, forget_bias=0.0, state_is_tuple=True)
        else:
            raise ValueError("Unknown rnn type")

        if is_training and config.keep_prob < 1:  # Dropout
            lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)

        cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, emb_size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
            # input_shape = input_.input_data.get_shape()
            # input_flatten = tf.reshape(input_.input_data, [-1, 1])
            # batch_size_flatten = tf.size(input_flatten)[0]
            # batch_size_flatten = int(input_shape[0]*input_shape[1])
            # indices = tf.expand_dims(tf.range(0, batch_size_flatten, 1), 1)
            # concated = tf.concat(1, [indices, input_flatten])
            # onehot_input = tf.sparse_to_dense(
            #    concated, tf.pack([batch_size_flatten, vocab_size]), 1.0, 0.0)
            # inputs = tf.reshape(tf.matmul(onehot_input, embedding), [int(input_shape[0]), int(input_shape[1]), emb_size])

            print("input to model: ", input_.input_data)
            print("input after embedding layer:  ", inputs)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = [tf.squeeze(input_step, [1])
        #           for input_step in tf.split(1, num_steps, inputs)]
        # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)

        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN") as scope:
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                with tf.variable_scope("softmax") as scope:
                    if FLAGS.output_type == "linear":
                        logits = rnn_cell._linear(cell_output, vocab_size, True, scope=scope)
                    elif FLAGS.output_type == "knet":
                        logits = rnn_cell._klinear_flat(cell_output, vocab_size, True, scope=scope)
                    else:
                        raise ValueError("Unknown output layer type")
                # print(logits)
                outputs.append(logits)
                # outputs.append(cell_output)

        logits = tf.reshape(tf.concat(axis=1, values=outputs), [-1, vocab_size])
        # output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])

        # with tf.variable_scope("softmax") as scope:
        #  if FLAGS.useKnetOutput:
        #    logits = rnn_cell._klinear_flat(output, vocab_size, True, scope = scope)
        #  else:
        #    logits = rnn_cell._linear(output, vocab_size, True, scope = scope)
        #    #softmax_w = tf.get_variable(
        #    #    "softmax_w", [size, vocab_size], dtype=data_type())
        #    #softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        #    #logits = tf.nn.bias_add(tf.matmul(output, softmax_w), softmax_b)

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(input_.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=data_type())])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()

        self._QR = None  # Todo:

        if FLAGS.useSGD:
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                              config.max_grad_norm)
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
            self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step())
        elif FLAGS.useAdam:
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                              config.max_grad_norm)
            optimizer = tf.train.AdamOptimizer(self._lr * 0.001)
            self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op
