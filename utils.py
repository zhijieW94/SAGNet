import tensorflow as tf
import yaml

def init_weights(shape, name):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def init_biases(shape):
    return tf.Variable(tf.zeros(shape))


def threshold(x, val=0.5):
    x = tf.clip_by_value(x, 0.5, 0.5001) - 0.5
    x = tf.minimum(x * 10000, 1)
    return x


def lrelu(x, leak=0.2, name=None):
    return tf.maximum(x, leak * x, name=name)


def make_var(name, shape=None, initializer=None, trainable=True):
    if shape is not None:
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    else:
        return tf.get_variable(name, initializer=initializer, trainable=trainable)


def exp_average_summary(ops, dep_ops, decay=0.9, name='avg', scope_pfix='', raw_pfix=' (raw)', avg_pfix=' (avg)'):
    averages = tf.train.ExponentialMovingAverage(decay, name=name)
    averages_op = averages.apply(ops)

    for op in ops:
        tf.summary.scalar(scope_pfix + op.name + raw_pfix, op)
        tf.summary.scalar(scope_pfix + op.name + avg_pfix, averages.average(op))

    with tf.control_dependencies([averages_op]):
        for i, dep_op in enumerate(dep_ops):
            dep_ops[i] = tf.identity(dep_op, name=dep_op.name.split(':')[0])

    return dep_ops


def load_from_yml(file_path):
    with open(file_path, 'r') as loadfile:
        y = yaml.load(loadfile)
        return y

def write_to_yml(data_dict, file_path):
    with open(file_path, 'w') as outfile:
        data_dict = dict(data_dict)
        yaml.dump(data_dict, outfile, default_flow_style=False)