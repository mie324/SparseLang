import numpy as np
import tensorflow as tf

def convert_to_sparse(labels, dtype=np.int32):
    indices = []
    values = []

    for n, seq in enumerate(labels):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=dtype)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(labels), np.asarray(indices).max(0)[1] + 1], dtype=dtype)

    # indices = tf.where(tf.not_equal(labels, tf.constant(0, dtype=tf.int32)))
    # values = tf.gather_nd(labels, indices)
    # sparse_labels = tf.SparseTensor(indices, values, dense_shape=tf.shape(labels, out_type=tf.int64))
    # print(sparse_labels)

    return indices, values, shape

def shrinkage_np(x, threshold=None, percent=0.1):
    # Only use when the matrix is a full matrix

    # Shrink the large matrix based on its value, keep larger value.
    # After some training , make the weght matrix sparse to test the performance.


    if threshold and percent:
        raise ValueError("threshold and percent can not be set simutaneously. Please only choose one instead")
    if threshold:  # Keep large value
        print("Keep value greater than {}".format(threshold))
        # mask1 = ma.masked_where(x > threshold, x)
        # mask2 = ma.masked_where(x < -threshold, x)
        mask1 = np.where(x > threshold, 1, 0)
        mask2 = np.where(x < -threshold, 1, 0)
        x = mask1 * x + mask2 * x
    if percent:  # Keep probability,keep large value
        print("Keep {}% of Matrix value".format(percent * 100))
        reshape_x = np.reshape(x, [-1])
        sort_value = -np.sort(-np.abs(reshape_x))
        threshold_value = sort_value[int(reshape_x.shape[0] * (percent))]  # Threshold keep probability
        # mask1 = ma.masked_where(x > threshold_value, x)
        # mask2 = ma.masked_where(x < -threshold_value, x)
        mask1 = np.where(x > threshold_value, 1, 0)
        mask2 = np.where(x < -threshold_value, 1, 0)
        x = mask1 * x + mask2 * x
    return x

def shrinkage_tf(x, threshold=None, percent=0.1):
    if threshold and percent:
        raise ValueError("threshold and percent can not be set simutaneously. Please only choose one instead")
    if threshold:  # Keep large value
        mask = tf.to_float(x > threshold)
        mask2 = tf.to_float(x < -threshold)
        x = mask * x + mask2 * x
    if percent:
        raise NotImplementedError
    return x

def dense_to_sparse_tf(dense_tensor, trainable=False, dtype=tf.float32):
    '''
    # Stupid Implementation
    :param dense_tensor: np array or tf.tensor with lots of 0.
    :return: tf.SparseTensor
    '''
    if trainable:
        # print("Create trainable tensor")
        dense_tensor = tf.Variable(dense_tensor, dtype=dtype)
    indices = tf.where(tf.not_equal(dense_tensor, tf.constant(0, dense_tensor.dtype)))
    values = tf.gather_nd(dense_tensor, indices)
    shape = tf.shape(dense_tensor, out_type=tf.int64)

    return tf.SparseTensor(indices, values, shape)