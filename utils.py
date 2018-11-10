import tensorflow as tf
import numpy as np
import matplotlib
from matplotlib.pyplot import imsave

FLAGS = tf.app.flags.FLAGS


def shrinkage(x, alpha=0.01):
    mask = tf.to_float(x > alpha)
    mask2 = tf.to_float(x < -alpha)
    x = mask * x + mask2 * x
    return x


def visualize(imgs, name):
    numImgs = len(imgs)
    stride = 2
    img_per_row = 10

    canvas = np.ones([(28 + stride) * img_per_row, (28 + stride) * img_per_row])
    for i in range(img_per_row):
        for j in range(img_per_row):
            idx_i = i * (28 + stride)
            idx_j = j * (28 + stride)
            if i * img_per_row + j < numImgs:
                canvas[idx_i:(idx_i + 28), idx_j:(idx_j + 28)] = imgs[i * img_per_row + j].reshape(28, 28)
            else:
                break

    imsave(name + '_' + FLAGS.name + '.png', canvas, cmap=matplotlib.cm.gray)


def gmatmul(a, b, transpose_a=False, transpose_b=False, reduce_dim=None):
    if reduce_dim == None:
        #### general batch matmul
        if len(a.get_shape()) == 3 and len(b.get_shape()) == 3:
            return tf.matmul(a, b, adj_x=transpose_a, adj_y=transpose_b)
        elif len(a.get_shape()) == 3 and len(b.get_shape()) == 2:
            if transpose_b:
                N = b.get_shape()[0].value
            else:
                N = b.get_shape()[1].value
            B = a.get_shape()[0].value
            if transpose_a:
                K = a.get_shape()[1].value
                a = tf.reshape(tf.transpose(a, [0, 2, 1]), [-1, K])
            else:
                K = a.get_shape()[-1].value
                a = tf.reshape(a, [-1, K])
            result = tf.matmul(a, b, transpose_b=transpose_b)
            result = tf.reshape(result, [B, -1, N])
            return result
        elif len(a.get_shape()) == 2 and len(b.get_shape()) == 3:
            if transpose_a:
                M = a.get_shape()[1].value
            else:
                M = a.get_shape()[0].value
            B = b.get_shape()[0].value
            if transpose_b:
                K = b.get_shape()[-1].value
                b = tf.transpose(tf.reshape(b, [-1, K]), [1, 0])
            else:
                K = b.get_shape()[1].value
                b = tf.transpose(tf.reshape(tf.transpose(b, [0, 2, 1]), [-1, K]), [1, 0])
            result = tf.matmul(a, b, transpose_a=transpose_a)
            result = tf.transpose(tf.reshape(result, [M, B, -1]), [1, 0, 2])
            return result
        else:
            return tf.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
    else:
        #### weird batch matmul
        if len(a.get_shape()) == 2 and len(b.get_shape()) > 2:
            ## reshape reduce_dim to the left most dim in b
            b_shape = b.get_shape()
            if reduce_dim != 0:
                b_dims = range(len(b_shape))
                b_dims.remove(reduce_dim)
                b_dims.insert(0, reduce_dim)
                b = tf.transpose(b, b_dims)
            a_t_shape = [item for item in a.get_shape()]
            b_t_shape = [item for item in b.get_shape()]
            b_t_shape[0] = a_t_shape[1] if transpose_a else a_t_shape[0]
            b_t_shape = tf.TensorShape(b_t_shape)
            b = tf.reshape(b, [int(b_shape[reduce_dim]), -1])
            result = tf.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
            result = tf.reshape(result, b_t_shape)

            if reduce_dim != 0:
                b_dims = range(len(b_shape))
                b_dims.remove(0)
                b_dims.insert(reduce_dim, 0)
                result = tf.transpose(result, b_dims)
            return result

        elif len(a.get_shape()) > 2 and len(b.get_shape()) == 2:
            ## reshape reduce_dim to the right most dim in a
            a_shape = a.get_shape()
            outter_dim = len(a_shape) - 1
            # reduce_dim = len(a_shape) - reduce_dim - 1
            if reduce_dim != outter_dim:
                a_dims = range(len(a_shape))
                a_dims.remove(reduce_dim)
                a_dims.insert(outter_dim, reduce_dim)
                a = tf.transpose(a, a_dims)
            a_t_shape = [item for item in a.get_shape()]
            b_t_shape = [item for item in b.get_shape()]
            a = tf.reshape(a, [-1, int(a_shape[reduce_dim])])
            a_t_shape[-1] = b_t_shape[0] if transpose_b else b_t_shape[1]
            a_t_shape = tf.TensorShape(a_t_shape)
            result = tf.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
            result = tf.reshape(result, a_t_shape)
            if reduce_dim != outter_dim:
                a_dims = range(len(a_shape))
                a_dims.remove(outter_dim)
                a_dims.insert(reduce_dim, outter_dim)
                result = tf.transpose(result, a_dims)
            return result

        elif len(a.get_shape()) == 2 and len(b.get_shape()) == 2:
            return tf.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)

        assert False, 'something went wrong'
