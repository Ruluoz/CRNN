

import tensorflow as tf
from tensorflow.contrib import rnn
from conv_attention import lstm_attention
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn

def Module_Attlstm(inputs, hidden_size, keep_prob, 
                   length,lstm_t=3):
                   
    att_ = lstm_attention(inputs) 
    input_ = tf.add_n([inputs,att_])
    layer = Bi_lstm(input_, hidden_size, keep_prob, length,)
        
    return layer


def Bi_lstm(input_, hidden_size, keep_prob, length):
    
    with tf.variable_scope("B_a1"):
        fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size/2, forget_bias=1.0,)
        fw_cell = rnn.DropoutWrapper(cell=fw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size/2, forget_bias=1.0)
        bw_cell = rnn.DropoutWrapper(cell=bw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        
        outputs1, output_states1 = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,
                                                                   inputs= input_,
                                                                   dtype=tf.float32,
                                                                   time_major=False,
                                                                   sequence_length=length,)
        
        state1 = tf.reduce_sum(outputs1[0], 1) / tf.reshape(tf.cast(length, dtype=tf.float32), [tf.shape(input_)[0], 1])
        state2 = tf.reduce_sum(outputs1[1], 1) / tf.reshape(tf.cast(length, dtype=tf.float32), [tf.shape(input_)[0], 1])
        state_1 = tf.concat([state1,state2], 1)
   
    return state_1

