import tensorflow as tf
import numpy as np
import sys

FLAGS = tf.flags.FLAGS
# Data file
# --input_file_pattern "/scratch/gobi1/zycluke/dataset/dataset_for_p_model/binarized/t_5000_d_10000/chunk/train-????"
tf.flags.DEFINE_string("input_file_pattern", None, "Input file pattern")
tf.flags.DEFINE_string("train_file", None, "Specify train file")
# --data_dir "/scratch/gobi1/zycluke/dataset/dataset_for_p_model/binarized/t_5000_d_10000/"
tf.flags.DEFINE_string("data_dir", None, "Data file directory.")
tf.flags.DEFINE_string("train_suffix", "train.tfrecord", "Train suffix")
tf.flags.DEFINE_string("val_suffix", "val.tfrecord", "val suffix")
tf.flags.DEFINE_string("test_suffix", "test.tfrecord", "test suffix")
tf.flags.DEFINE_string("inference_output_file", None, "Output file to store decoding results.")
tf.flags.DEFINE_string("inference_ref_file", None, "Reference file to compute evaluation scores (if provided).")
# Input specification
tf.flags.DEFINE_integer("batch_size", 128, "Batch size.")
tf.flags.DEFINE_integer("buffer_size", 30, "Buffer size.")
tf.flags.DEFINE_integer("prefetch_buffer_size", None, "Prefetch buffer size.")
tf.flags.DEFINE_integer("num_parallel_calls", 4, "Number of parallel calls.")
tf.flags.DEFINE_integer("num_buckets", 5, "Put data into similar-length buckets.")
tf.flags.DEFINE_integer("num_epochs", 1000, "Number of epochs.")
tf.flags.DEFINE_integer("infer_batch_size", 32, "Batch size for inference mode.")
tf.flags.DEFINE_integer("src_max_len", 5, "Max length of src sequences during training.")
tf.flags.DEFINE_integer("tgt_max_len", 50, "Max length of tgt sequences during training.")
tf.flags.DEFINE_integer("thought_vector_dim", 2400, "Thought vector dimension.")
tf.flags.DEFINE_integer("repeat_epoch", 100000, "number of epochs per evaluation.")
# Misc
tf.flags.DEFINE_integer("random_seed", None, "Random seed (>0, set a specific seed).")
tf.flags.DEFINE_integer("num_parallel_reads", 4, "Number of num_parallel_reads")
tf.flags.DEFINE_integer("take_dataset", -1, "how many example to take")


def parse_exmp(serial_exmp, START_VECTOR, END_VECTOR):
    features = tf.parse_single_example(serial_exmp, features={'source': tf.VarLenFeature(tf.float32),
                                                              'target': tf.VarLenFeature(tf.float32),
                                                              'src_seq_len': tf.FixedLenFeature([], tf.int64),
                                                              'tgt_seq_len': tf.FixedLenFeature([], tf.int64)})
    source = tf.sparse_tensor_to_dense(features['source'], default_value=0)
    target = tf.sparse_tensor_to_dense(features['target'], default_value=0)

    source = tf.cast(tf.reshape(source, (-1, FLAGS.thought_vector_dim)), tf.float32)
    target = tf.cast(tf.reshape(target, (-1, FLAGS.thought_vector_dim)), tf.float32)

    src_seq_len = tf.cast(features['src_seq_len'], tf.int32)
    tgt_seq_len = tf.cast(features['tgt_seq_len'], tf.int32)

    if FLAGS.src_max_len:
        src_seq_len = tf.minimum(src_seq_len, FLAGS.src_max_len)
        source = tf.slice(source, [0, 0], [src_seq_len, FLAGS.thought_vector_dim])
    if FLAGS.tgt_max_len:
        tgt_seq_len = tf.minimum(tgt_seq_len, FLAGS.tgt_max_len)
        target = tf.slice(target, [0, 0], [tgt_seq_len, FLAGS.thought_vector_dim])

    target_in = tf.concat(([START_VECTOR], target), 0)
    target_out = tf.concat((target, [END_VECTOR]), 0)

    tgt_in_seq_len = tgt_seq_len + 1

    return source, target, target_in, target_out, src_seq_len, tgt_seq_len, tgt_in_seq_len


def bucket_batch(dataset, FLAGS):
    def batching_func(x):
        batch_size = int(FLAGS.batch_size / FLAGS.num_gpus)
        return x.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None, 2400]),  # src
                                                         tf.TensorShape([None, 2400]),  # tgt
                                                         tf.TensorShape([None, 2400]),  # tgt_input
                                                         tf.TensorShape([None, 2400]),  # tgt_output
                                                         tf.TensorShape([]),  # src_len
                                                         tf.TensorShape([]),
                                                         tf.TensorShape([])),  # tgt_len
                              padding_values=(0.0,  # src
                                              0.0,  # tgt
                                              0.0,  # tgt_input
                                              0.0,  # tgt_output
                                              0,  # src_len
                                              0,
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


def output_mapping(source, target, target_in, target_out, src_seq_len, tgt_seq_len, tgt_in_seq_len):
    example = {}
    example['source'] = source
    example['target'] = target
    example['target_in'] = target_in
    example['target_out'] = target_out
    example['src_seq_len'] = src_seq_len
    example['tgt_seq_len'] = tgt_seq_len
    example['tgt_in_seq_len'] = tgt_in_seq_len

    label = tf.constant(1, dtype=tf.int32)

    return example, label


def train_input_fn():
    END_VECTOR = tf.ones(shape=FLAGS.thought_vector_dim, dtype=tf.float32)
    START_VECTOR = tf.zeros(shape=FLAGS.thought_vector_dim, dtype=tf.float32)

    print("\n input_file_pattern", FLAGS.input_file_pattern)
    print(FLAGS.flag_values_dict())

    if FLAGS.train_file:
        train_datafile = FLAGS.train_file
    elif FLAGS.input_file_pattern:
        train_datafile = tf.data.Dataset.list_files(file_pattern=FLAGS.input_file_pattern)
        print('Create Dataset from input file pattern:', FLAGS.input_file_pattern)
    else:
        if not FLAGS.data_dir:
            raise ValueError("train_file, data_dir or input_file_pattern is required.")
        train_datafile = FLAGS.data_dir + FLAGS.train_suffix

    print('datafile dirctory:', train_datafile)

    dataset = tf.data.TFRecordDataset(train_datafile, num_parallel_reads=FLAGS.num_parallel_reads)
    if FLAGS.take_dataset:
        print(FLAGS.take_dataset)
        dataset = dataset.take(FLAGS.take_dataset)
    dataset = dataset.apply(
        tf.contrib.data.shuffle_and_repeat(buffer_size=FLAGS.buffer_size, count=FLAGS.repeat_epoch)).prefetch(
        buffer_size=FLAGS.buffer_size)
    dataset = dataset.map(map_func=lambda serial_exmp: parse_exmp(serial_exmp, START_VECTOR, END_VECTOR),
                          num_parallel_calls=FLAGS.num_parallel_calls).prefetch(
        buffer_size=FLAGS.buffer_size)
    dataset = bucket_batch(dataset, FLAGS).prefetch(
        buffer_size=FLAGS.prefetch_buffer_size)
    dataset = dataset.map(map_func=output_mapping, num_parallel_calls=FLAGS.num_parallel_calls).prefetch(
        buffer_size=FLAGS.prefetch_buffer_size)

    return dataset


# from tensorflow.contrib.data.python.ops import shuffle_ops
def eval_input_fn():
    END_VECTOR = tf.ones(shape=FLAGS.thought_vector_dim, dtype=tf.float32)
    START_VECTOR = tf.zeros(shape=END_VECTOR.shape, dtype=tf.float32)

    if FLAGS.input_file_pattern:
        eval_datafile = tf.data.Dataset.list_files(file_pattern=FLAGS.input_file_pattern)
        print('Create Dataset from input file pattern:', FLAGS.input_file_pattern)
    else:
        if not FLAGS.data_dir:
            raise ValueError("data_dir or input_file_pattern is required.")
        eval_datafile = FLAGS.data_dir + FLAGS.val_suffix
        print('datafile dirctory:', eval_datafile)

    dataset = tf.data.TFRecordDataset(eval_datafile)
    if FLAGS.take_dataset:
        print(FLAGS.take_dataset)
        dataset = dataset.take(FLAGS.take_dataset)
    dataset = dataset.shuffle(buffer_size=FLAGS.buffer_size, seed=FLAGS.random_seed)
    dataset = dataset.map(map_func=lambda serial_exmp: parse_exmp(serial_exmp, START_VECTOR, END_VECTOR),
                          num_parallel_calls=FLAGS.num_parallel_calls).prefetch(
        buffer_size=FLAGS.buffer_size)
    dataset = bucket_batch(dataset, FLAGS).prefetch(
        buffer_size=FLAGS.prefetch_buffer_size)
    dataset = dataset.map(map_func=output_mapping, num_parallel_calls=FLAGS.num_parallel_calls).prefetch(
        buffer_size=FLAGS.prefetch_buffer_size)

    return dataset


def small_train_input():
    END_VECTOR = tf.ones(shape=FLAGS.thought_vector_dim, dtype=tf.float32)
    START_VECTOR = tf.zeros(shape=END_VECTOR.shape, dtype=tf.float32)

    if FLAGS.train_file:
        train_datafile = FLAGS.train_file
    elif FLAGS.input_file_pattern:
        train_datafile = tf.data.Dataset.list_files(file_pattern=FLAGS.input_file_pattern)
        print('Create Dataset from input file pattern:', FLAGS.input_file_pattern)
    else:
        if not FLAGS.data_dir:
            raise ValueError("train_file, data_dir or input_file_pattern is required.")
        train_datafile = FLAGS.data_dir + FLAGS.train_suffix
        print('datafile dirctory:', train_datafile)

    dataset = tf.data.TFRecordDataset(train_datafile)
    if FLAGS.take_dataset:
        print(FLAGS.take_dataset)
        dataset = dataset.take(FLAGS.take_dataset)
    dataset = dataset.repeat()
    dataset = dataset.map(map_func=lambda serial_exmp: parse_exmp(serial_exmp, START_VECTOR, END_VECTOR),
                          num_parallel_calls=FLAGS.num_parallel_calls).prefetch(
        buffer_size=FLAGS.buffer_size)
    dataset = bucket_batch(dataset, FLAGS).prefetch(
        buffer_size=FLAGS.prefetch_buffer_size)
    dataset = dataset.map(map_func=output_mapping, num_parallel_calls=FLAGS.num_parallel_calls).prefetch(
        buffer_size=FLAGS.prefetch_buffer_size)

    return dataset


def main(unused_argv):
    # Testing function
    print('Hello world')
    exmp = train_input_fn()
    iterator = exmp.make_one_shot_iterator()
    exmp, label = iterator.get_next()

    print('Start_session!')

    with tf.Session() as sess:
        # sess.run(iterator.initializer)
        problemindex = []

        for i in range(3000):
            try:
                cur_ex = sess.run([exmp])[0]

                if i % 20 == 0:
                    print(i)
                    sys.stdout.flush()

                src_seq_len = cur_ex['src_seq_len']
                tgt_seq_len = cur_ex['tgt_seq_len']

                if src_seq_len.shape != tgt_seq_len.shape:
                    problemindex.append(i)
                    print('src_seq_len', src_seq_len)
                    print('tgt_seq_len', tgt_seq_len)

                if i > 100 and i % 20 == 0:
                    print(i)
                    print('source', np.array(cur_ex['source']).shape)
                    print('target', np.array(cur_ex['target']).shape)
                    print('target_in', np.array(cur_ex['target_in']).shape)
                    print('target_out', np.array(cur_ex['target_out']).shape)
                    print('src_seq_len', np.array(cur_ex['src_seq_len']).shape)
                    print('tgt_seq_len', np.array(cur_ex['tgt_seq_len']).shape)
                    print('tgt_in_seq_len', np.array(cur_ex['tgt_in_seq_len']).shape)


            except tf.errors.OutOfRangeError:
                print("End of dataset")
                print('current ', i)
                break
                # sess.run(iterator.initializer)

        if problemindex:
            print(problemindex)

    print('finished!')


if __name__ == '__main__':
    tf.app.run(main)
