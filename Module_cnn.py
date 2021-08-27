
import tensorflow as tf
import numpy as np
from conv_attention import cnn_attention

class Module_cnn(object):
    """Implementation of the AlexNet."""

    def __init__(self, x,  size, skip_layer,):
        
        self.X = x
        self.size = int(size)
        self.SKIP_LAYER = skip_layer   
        self.WEIGHTS_PATH = 'bvlc_alexnet.npy'

        self.create()
        
    def create(self):
        
        conv1 = conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 2, 2, 2, 2, padding='VALID', name='pool1')

        L1 = cnn_attention(pool1, filter2_size=self.size, layer_name='att_cnn1') 
        
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 2, 2, 2, 2, padding='VALID', name='pool2')

        L2 = cnn_attention(pool2, filter2_size=self.size, layer_name='att_cnn2')

        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 2, 2, 2, 2, padding='VALID', name='pool5')
      
        L3 = cnn_attention(pool5, filter2_size=self.size, layer_name='att_cnn3')   
                
        self.outpus = tf.add_n([L1, L2, L3])
                
            
    def load_initial_weights(self, session):

        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes',allow_pickle=True).item()

        for op_name in weights_dict:  
            if op_name not in self.SKIP_LAYER:
                with tf.variable_scope(op_name, reuse=True):
                    for data in weights_dict[op_name]:
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))



def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    
    input_channels = int(x.get_shape()[-1])

    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    else:
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        conv = tf.concat(axis=3, values=output_groups)

    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc(x, num_in, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):

    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):

    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)






