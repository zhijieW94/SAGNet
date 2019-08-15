import numpy as np
import tensorflow as tf
from utils import *

DEFAULT_PADDING = 'SAME'

"""
A wrapper for TensorFlow network layers
Adapted from SubCNN_TF (https://github.com/yuxng/SubCNN_TF)
"""

def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs) == 0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs) == 1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        if name in self.layers:
            print('overriding layer %s!!!!' % name)
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            # We get the right layers corresponding to the names in the list
            if isinstance(layer, basestring):
                try:
                    layer = self.layers[layer]
                    print "feed function"
                    print layer
                except KeyError:
                    print "feed function"
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def wrap(self, tensor, name):
        self.layers[name] = tensor
        return self.feed(tensor)

    @layer
    def stop_gradient(self, input, name):
        return tf.stop_gradient(input, name=name)

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print "get_output function"
            print self.layers.keys()
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def activation_summary(self, layer_name):
        try:
            layer = self.layers[layer_name]
            self._variable_summaries(layer, layer_name)

        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def _variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            tf.summary.histogram(name, var)

    def get_unique_name(self, prefix):
        # in this function, we count the number of certain kind of layers
        id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    def conv_residual_block(self, input_layer, k_size, output_channel, init_weight, name=None, leaky_value=0.0,
                            padding='SAME', trainable=True, reuse=False, bottle_neck=False):
        input_channel = input_layer.get_shape().as_list()[-1]

        # if input_channel * 2 != output_channel or input_channel != output_channel:
        #     raise ValueError("Output and input channel does not match in convolution residual blocks!!!")

        if not (input_channel * 2 == output_channel or input_channel == output_channel) == True:
            raise ValueError("Output and input channel does not match in convolution residual blocks!!!")

        if bottle_neck:
            valid_strides = [1, 2, 2, 2, 1]
        else:
            valid_strides = [1, 1, 1, 1, 1]

        same_strides = [1, 1, 1, 1, 1]

        with tf.variable_scope(name) as scope:
            if reuse: tf.get_variable_scope().reuse_variables()

            short_cut = input_layer
            if bottle_neck:
                left_weight_name = 'left_conv'
                left_kernel = make_var(left_weight_name, [1, 1, 1, input_channel, output_channel], init_weight, trainable=trainable)
                left_conv_out = tf.nn.conv3d(input_layer, left_kernel, strides=valid_strides, padding=padding, name='left_conv')
                short_cut = left_conv_out

            right_first_depth = output_channel / 2
            right_second_depth = output_channel / 2
            right_third_depth = output_channel

            right_first_layer_name = 'right_conv_0'
            right_second_layer_name = 'right_conv_1'
            right_third_layer_name = 'right_conv_2'

            right_first_conv = self.conv3d(input_layer, 1, right_first_depth, right_first_layer_name, init_weight,
                                           strides=valid_strides, padding=DEFAULT_PADDING, leaky_value=leaky_value,
                                           relu=True, has_biases=False, batch_norm=True, trainable=trainable, reuse=reuse)

            right_second_conv = self.conv3d(right_first_conv, k_size, right_second_depth, right_second_layer_name, init_weight,
                                            strides=same_strides, padding=DEFAULT_PADDING, leaky_value=leaky_value, relu=True,
                                            has_biases=False, batch_norm=True, trainable=trainable, reuse=reuse)

            right_third_conv = self.conv3d(right_second_conv, 1, right_third_depth, right_third_layer_name, init_weight,
                                           strides=same_strides, padding=DEFAULT_PADDING, leaky_value=leaky_value, relu=False,
                                           has_biases=False, batch_norm=True, trainable=trainable, reuse=reuse)

            res_block_out = tf.add(short_cut, right_third_conv)
            res_block_out = lrelu(res_block_out, leaky_value, name=scope.name)
            return res_block_out


    def deconv_residual_block(self, input_layer, k_size, output_channel, target_shape, init_weights, name=None, leaky_value=0.0,
                              padding='SAME', trainable=True, reuse=False, bottle_neck=True):
        input_channel = input_layer.get_shape().as_list()[-1]

        # if input_channel != output_channel * 2 or input_channel != output_channel:
        #     raise ValueError("Output and input channel does not match in deconvolution residual blocks!!!")

        if not (input_channel == output_channel * 2 or input_channel == output_channel) == True:
            raise ValueError("Output and input channel does not match in deconvolution residual blocks!!!")

        if bottle_neck:
            valid_strides = [1, 2, 2, 2, 1]
        else:
            valid_strides = [1, 1, 1, 1, 1]
        same_strides = [1, 1, 1, 1, 1]

        with tf.variable_scope(name) as scope:
            if reuse: tf.get_variable_scope().reuse_variables()

            short_cut = input_layer
            if bottle_neck:
                left_weight_name = 'left_deconv'
                left_deconv_out = self.deconv3d(input_layer, 1, output_channel, left_weight_name, target_shape, init_weights, strides=valid_strides,
                                                leaky_value=leaky_value, padding=DEFAULT_PADDING, relu=False,
                                                has_biases=False, batch_norm=True, trainable=trainable, reuse=reuse)
                short_cut = left_deconv_out

            right_first_depth = output_channel / 2
            right_second_depth = output_channel / 2
            right_third_depth = output_channel

            right_first_layer_name = 'right_deconv_0'
            right_second_layer_name = 'right_deconv_1'
            right_third_layer_name = 'right_deconv_2'

            first_target_shape = [target_shape[0], target_shape[1], target_shape[2], target_shape[3], right_first_depth]
            second_target_shape = [target_shape[0], target_shape[1], target_shape[2], target_shape[3], right_second_depth]
            third_target_shape = [target_shape[0], target_shape[1], target_shape[2], target_shape[3], right_third_depth]

            right_first_deconv = self.deconv3d(input_layer, 1, right_first_depth, right_first_layer_name,
                                             first_target_shape, init_weights, strides=valid_strides,
                                             leaky_value=leaky_value, padding=DEFAULT_PADDING, relu=True,
                                             has_biases=False, batch_norm=True, trainable=trainable, reuse=reuse)

            right_second_deconv = self.deconv3d(right_first_deconv, k_size, right_second_depth, right_second_layer_name,
                                                second_target_shape, init_weights, strides=same_strides,
                                             leaky_value=leaky_value, padding=DEFAULT_PADDING, relu=True,
                                             has_biases=False, batch_norm=True, trainable=trainable, reuse=reuse)

            right_third_deconv = self.deconv3d(right_second_deconv, 1, right_third_depth, right_third_layer_name,
                                               third_target_shape, init_weights, strides=same_strides,
                                             leaky_value=leaky_value, padding=DEFAULT_PADDING, relu=False,
                                             has_biases=False, batch_norm=True, trainable=trainable, reuse=reuse)

            res_block_out = tf.add(short_cut, right_third_deconv)
            res_block_out = lrelu(res_block_out, leaky_value, name=scope.name)
            return res_block_out


    # @layer
    def conv3d(self, input, k_size, out_depth, name, init_weights, strides, init_biases=None, padding=DEFAULT_PADDING, leaky_value=0.0,
               relu=True, has_biases=True, batch_norm=True, trainable=True, reuse=False):
        self.validate_padding(padding)
        weight_name = 'weight'
        biases_name = 'biases'
        in_depth = input.get_shape()[-1]
        with tf.variable_scope(name) as scope:
            if reuse: tf.get_variable_scope().reuse_variables()

            # kernel = make_var(weight_name, [k_size, k_size, k_size, in_depth, out_depth], init_weights, trainable=trainable)
            # biases = make_var(biases_name, [out_depth], init_biases, trainable=trainable)
            # conv3d = tf.nn.conv3d(input, kernel, strides=strides, padding=padding, name=name)
            # bias_out = tf.nn.bias_add(conv3d, biases)
            #
            # conv_out = bias_out

            if has_biases:
                kernel = make_var(weight_name, [k_size, k_size, k_size, in_depth, out_depth], init_weights, trainable=trainable)
                biases = make_var(biases_name, [out_depth], init_biases, trainable=trainable)
                conv3d_out = tf.nn.conv3d(input, kernel, strides=strides, padding=padding, name=name)
                conv_out = tf.nn.bias_add(conv3d_out, biases)

            else:
                kernel = make_var(weight_name, [k_size, k_size, k_size, in_depth, out_depth], init_weights, trainable=trainable)
                conv_out = tf.nn.conv3d(input, kernel, strides=strides, padding=padding, name=name)

            if batch_norm:
                conv_out = tf.contrib.layers.batch_norm(conv_out, is_training=trainable)
            if relu:
                conv_out = lrelu(conv_out, leaky_value, scope.name)
            return conv_out

    # @layer
    def deconv3d(self, input, k_size, out_depth, name, target_shape, init_weights, strides, init_biases=None, leaky_value=0.0, relu=True, batch_norm=True,
                 padding=DEFAULT_PADDING, has_biases=False, trainable=True, reuse=False):
        self.validate_padding(padding)
        weight_name = 'weight'
        biases_name = 'biases'
        in_depth = input.get_shape()[-1]
        with tf.variable_scope(name) as scope:
            if reuse: tf.get_variable_scope().reuse_variables()
            # kernel = make_var(weight_name, [k_size, k_size, k_size, out_depth, in_depth], init_weights, trainable=trainable)
            # biases = make_var(biases_name, [out_depth], init_biases, trainable=trainable)
            #
            # deconv3d = tf.nn.conv3d_transpose(input, kernel, target_shape, strides=strides, padding=padding, name=name)
            # bias_out = tf.nn.bias_add(deconv3d, biases)
            #
            # deconv_out = bias_out

            if has_biases:
                kernel = make_var(weight_name, [k_size, k_size, k_size, out_depth, in_depth], init_weights, trainable=trainable)
                biases = make_var(biases_name, [out_depth], init_biases, trainable=trainable)

                deconv3d_out = tf.nn.conv3d_transpose(input, kernel, target_shape, strides=strides, padding=padding, name=name)
                deconv_out = tf.nn.bias_add(deconv3d_out, biases)
            else:
                kernel = make_var(weight_name, [k_size, k_size, k_size, out_depth, in_depth], init_weights, trainable=trainable)
                deconv_out = tf.nn.conv3d_transpose(input, kernel, target_shape, strides=strides, padding=padding, name=name)

            if batch_norm:
                deconv_out = tf.contrib.layers.batch_norm(deconv_out, is_training=trainable)
            if relu:
                # deconv_out = tf.nn.relu(deconv_out, scope.name)
                deconv_out = lrelu(deconv_out, leaky_value, name=scope.name)
            return deconv_out

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def sigmoid(self, input, name):
        return tf.sigmoid(input, name=name)

    @layer
    def tanh(self, input, name):
        return tf.tanh(input, name=name)

    @layer
    def lrelu(self, input, leaky_value, name):
        return lrelu(input, leaky_value, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        # return tf.concat(concat_dim=axis, values=inputs, name=name)
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True, leaky_value = 0.0, trainable=True, reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            print name
            print input_shape
            if input_shape.ndims == 5:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            elif input_shape.ndims == 3:
                dim = 1
                for d in input_shape[2:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
            init_biases = tf.constant_initializer(0.0)
            weights = make_var('weights', [dim, num_out], init_weights, trainable)
            biases = make_var('biases', [num_out], init_biases, trainable)

            if relu:
                # fc = tf.nn.relu(tf.matmul(feed_in, weights) + biases, name=scope.name)
                fc = lrelu(tf.matmul(feed_in, weights) + biases, leaky_value, scope.name)
            else:
                fc = tf.add(tf.matmul(feed_in, weights), biases, name=scope.name)

            return fc

    @layer
    def softmax(self, input, name):
        input = tf.cast(input, dtype=tf.float64)
        return tf.nn.softmax(input, name=name)

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)

    @layer
    def argmax(self, input, dim, name):
        return tf.argmax(input, dim, name=name)

    @layer
    def identity(self, input, name):
        return tf.identity(input, name=name)

    @layer
    def batch_norm(self, input, name, is_training=True, relu=False, reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()

            t = tf.contrib.layers.batch_norm(input, is_training=is_training)
            if relu:
                t = tf.nn.relu(t)
            return t
