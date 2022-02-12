#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    if downsample:
        # ((top_pad, bottom_pad), (left_pad, right_pad))
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'
    
    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)
    
    if bn: conv = BatchNormalization()(conv)
    if activate:
        if activate_type == 'leaky':
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == 'mish':
            conv = mish(conv)
    return conv

def mish(x):
    return x*tf.math.tanh(tf.math.softplus(x))

def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = convolutional(conv,        filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)

    residual_block = short_cut + conv
    return residual_block

def cross_stage_partial(input_data, input_channel, filter_num1, filter_num2, filter_num3, res_num, activate_type_):
    route = input_data
    route    = convolutional(route,      filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type_)
    csp_data = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type_)
    for i in range(res_num):
        csp_data = residual_block(csp_data, filter_num1, filter_num2, filter_num3, activate_type=activate_type_)
    csp_data = convolutional(csp_data, filters_shape=(1, 1, filter_num3, filter_num3), activate_type=activate_type_)
    
    csp_data = tf.concat([csp_data, route], axis=-1)
    csp_data = convolutional(csp_data, filters_shape=(1, 1, filter_num1+filter_num3, input_channel), 
                               activate_type=activate_type_)
    return csp_data

def spp_block(input_data, input_channel):
    spp_data = tf.concat([tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1), 
                          tf.nn.max_pool(input_data, ksize=9,  padding='SAME', strides=1),
                          tf.nn.max_pool(input_data, ksize=5,  padding='SAME', strides=1),
                          input_data], axis=-1)
    return spp_data

def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1]*2, input_layer.shape[2]*2), method='nearest')

def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]