
import tensorflow as tf
import numpy as np

def BN(inputs, train_, layer_name="att2"):
    
#    parameter = inputs.get_shape().as_list()[1]
    parameter = 1024
         
    with tf.variable_scope(layer_name):
        
        pop_mean = tf.Variable(tf.zeros([parameter]), trainable=False)
        pop_variance = tf.Variable(tf.ones([parameter]), trainable=False)        
    
        axis = list(range(1))
        wb_mean, wb_var = tf.nn.moments(inputs, axis)
    
        decay = 0.99
        train_mean = tf.assign(pop_mean, pop_mean*decay + wb_mean*(1 - decay))
        train_variance = tf.assign(pop_variance, pop_variance*decay + wb_var*(1 - decay))
    
        scale = tf.Variable(tf.ones([parameter]))
        offset = tf.Variable(tf.zeros([parameter]))
        variance_epsilon = 0.001                    
               
        out = tf.cond(train_,lambda:tf.nn.batch_normalization(inputs, wb_mean, wb_var, offset, scale, variance_epsilon),
                           lambda:tf.nn.batch_normalization(inputs, train_mean, train_variance, offset, scale, variance_epsilon))
        
    return out


def batch_iter(Data_1, Data_2, label, length, batch_size, shuffle = True):
       
    data_size = len(Data_1)
    if len(Data_1) % batch_size == 0:
        num_batches_per_epoch = int(len(Data_1) / batch_size)
    else:
        num_batches_per_epoch = int(len(Data_1) / batch_size) + 1
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data_1 = Data_1[shuffle_indices]
        shuffled_data_2 = Data_2[shuffle_indices]
        shuffled_label = label[shuffle_indices]
        shuffled_length = length[shuffle_indices]
    else:
        shuffled_data_1 = Data_1
        shuffled_data_2 = Data_2
        shuffled_label = label
        shuffled_length = length
    for batch_num in range(num_batches_per_epoch):
        start = batch_num * batch_size
        end = min((batch_num + 1) * batch_size, data_size)
 
        yield shuffled_data_1[start:end], shuffled_data_2[start:end], shuffled_label[start:end], shuffled_length[start:end]
                
        
