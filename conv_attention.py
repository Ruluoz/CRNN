
import tensorflow as tf

def cnn_attention(inputs, filter2_size, att=True, layer_name='att_cnn', if_sum=True):

    if inputs.shape.ndims == 3:
        inputs = tf.expand_dims(inputs, -1)
        
    frame_size = inputs.shape[1].value
    feature_size = inputs.shape[2].value
    kernel_size = inputs.shape[3].value
    
    with tf.variable_scope(layer_name):
        
        if att == True:
            att_size = feature_size
        else:
            att_size = 1024
       
        att_filter1 = tf.Variable(tf.truncated_normal([1, 1, kernel_size, 1],stddev=0.01))
      
        att_conv1 = tf.nn.conv2d(inputs, filter=att_filter1, strides=[1, 1, 1, 1], padding='SAME')
        att_conv1 = tf.nn.relu(att_conv1) 
        squeeze_layer = tf.reshape(att_conv1, [-1, frame_size, feature_size])  
        
        att_frame_parameter = attention(squeeze_layer, att_size, return_alphas=True, layer_name=layer_name)

        att_filter2 = tf.Variable(tf.truncated_normal([1, 1, kernel_size, filter2_size],stddev=0.01))
       
        att_conv2 = tf.nn.conv2d(inputs, filter=att_filter2, strides=[1, 1, 1, 1], padding='SAME')
        att_conv2 = tf.nn.relu(att_conv2)
        pool_frame = tf.reduce_max(att_conv2, axis=2) 

        att_frame = pool_frame * tf.expand_dims(att_frame_parameter, -1)
        
        if if_sum == True:
            return tf.reduce_sum(att_frame, 1)
        else:
            return att_frame 

    return att_frame 


def lstm_attention(inputs, layer_name='att_lstm'):
    
    if inputs.shape.ndims == 3:
        inputs = tf.expand_dims(inputs, -1)
    
    frame_size = inputs.shape[1].value
    feature_size = inputs.shape[2].value
    kernel_size = inputs.shape[3].value
    
    with tf.variable_scope(layer_name):
        
        att_size = feature_size

        a_filter = tf.Variable(tf.truncated_normal([1, 1, kernel_size, 1],stddev=0.01))
        a_conv = tf.nn.conv2d(inputs, filter=a_filter, strides=[1, 1, 1, 1], padding='SAME')
        a_conv = tf.nn.relu(a_conv) 

        a_reshape = tf.reshape(a_conv, [-1, frame_size, feature_size]) 
        
        a_att= attention(a_reshape, att_size, return_alphas=True, layer_name='att_lstm1')  
       
        att_frame1 = a_reshape * tf.expand_dims(a_att, -1)
                
        inputs_ =  tf.transpose(inputs, [0, 2, 1, 3])

        b_filter = tf.Variable(tf.truncated_normal([1, 1, kernel_size, 1],stddev=0.01))
        b_conv = tf.nn.conv2d(inputs_, filter=b_filter, strides=[1, 1, 1, 1], padding='SAME')
        b_conv = tf.nn.relu(b_conv) 

        b_reshape = tf.reshape(b_conv, [-1, feature_size, frame_size]) 

        b_att= attention(b_reshape, att_size, return_alphas=True, layer_name='att_lstm2')

        att_frame2 = b_reshape * tf.expand_dims(b_att, -1)
                     
        att_frame2 = tf.transpose(att_frame2, [0, 2, 1])
            
        att_frame = tf.add_n([att_frame1,att_frame2])
        
    return att_frame


def attention(inputs, attention_size, return_alphas=True, layer_name="att"):

    hidden_size = inputs.shape[2].value  
    
    with tf.variable_scope(layer_name):
        
        W_omega = tf.Variable(tf.truncated_normal([hidden_size, attention_size], stddev=0.01))
        b_omega = tf.Variable(tf.truncated_normal([attention_size], stddev=0.01))
        u_omega = tf.Variable(tf.truncated_normal([attention_size], stddev=0.01))

        v = tf.sigmoid(tf.tensordot(inputs, W_omega, axes=1) + b_omega)
        
        vu = tf.tensordot(v, u_omega, axes=1)   # (B,T) shape
        alphas = tf.nn.softmax(vu)              # (B,T) shape also

        output = inputs * tf.expand_dims(alphas, -1)

    if return_alphas == False:
        return output
    else:
        return alphas        
  
        
        
        