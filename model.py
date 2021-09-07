
import tensorflow as tf
import numpy as np
from Module_cnn import Module_cnn
from Module_attlstm import Module_Attlstm
from tensorflow.python.ops import math_ops
from utils import BN, batch_iter
from sklearn.metrics import recall_score

def Model(x_train_cnn, x_train_rnn, 
          y_train, 
          lengths_train, 
          x_test_cnn, 
          x_test_rnn, 
          y_test, 
          lengths_test, 
          lenh,
          mood):

    epoch_times = 200
    hidden_num = 1024
    batch_train = 64

    x_cnn = tf.placeholder(tf.float32, [None, len(x_train_cnn[0][0]), len(x_train_cnn[0]), 3])
    x_rnn = tf.placeholder(tf.float32, [None, lenh, len(x_train_rnn[0][0])])
    y = tf.placeholder(tf.float32, [None, mood])
    batch_size = tf.placeholder(tf.int32, [])
    keep_prob = tf.placeholder(tf.float32)
    length = tf.placeholder(tf.int32, [None])
    train_ = tf.placeholder(dtype=tf.bool)
               

    init_cnn = Module_cnn(x_cnn, keep_prob, hidden_num, ['fc6','fc7','fc8'])
    out_cnn = init_cnn.outpus
    model_cnn = BN(out_cnn, train_,'BN_cnn')     
    model_cnn=math_ops.sigmoid(model_cnn)
      
    out_lstm = Module_Attlstm(x_rnn, hidden_num, keep_prob, length)
    out_lstm1 = BN(out_lstm, train_,'BN_LSTM')                
    model_attlstm = tf.nn.relu(out_lstm1)
       
    module = tf.concat([model_cnn, model_attlstm],1)                

    att_f= tf.get_variable("att_f",[hidden_num*2,hidden_num*2], initializer = tf.contrib.layers.xavier_initializer())
    att_v= tf.get_variable("att_v",[hidden_num*2,hidden_num*2], initializer = tf.contrib.layers.xavier_initializer())    
    module= tf.matmul(math_ops.sigmoid(tf.matmul(module, att_f)), att_v)
            
    W = tf.Variable(tf.truncated_normal(shape=[1024, mood], stddev=0.1), dtype=tf.float32)
    bias = tf.Variable(tf.constant(0.01,shape=[mood]), dtype=tf.float32)
    y_pre = tf.matmul(module , W) + bias

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_pre))         

    train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)
    
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                   
    init = tf.global_variables_initializer()
	  
    batch_test = int(len(y_test)) 
    epoch_ss = int(len(y_train)/batch_train)

    with tf.Session() as sess:     
        sess.run(init)
        init_cnn.load_initial_weights(sess)
        WA = []
        UA = []
        steps = 0
        epoch = 0
        
        while True: 
            for batch_cnn, batch_rnn, batch_ys, batch_len in batch_iter(x_train_cnn, x_train_rnn, y_train, lengths_train, batch_train,):                   
                steps += 1                                       
                sess.run(train_op, feed_dict={x_cnn: batch_cnn, x_rnn: batch_rnn, y: batch_ys, length: batch_len, keep_prob: 0.7, batch_size: batch_train, train_ : True})  

                if steps % epoch_ss == 0:                      
                    epoch += 1
  
                    wa = sess.run(accuracy, feed_dict={x_cnn: x_test_cnn, x_rnn: x_test_rnn, y: y_test, length: lengths_test, keep_prob: 1.0, batch_size: batch_test, train_ : False})
                    WA.append(wa)
                           
                    ff = sess.run(tf.argmax(y_pre,1),feed_dict={x_cnn: x_test_cnn, x_rnn: x_test_rnn, y: y_test, length: lengths_test, keep_prob: 1.0, batch_size: batch_test, train_ : False})
                    fg = sess.run(tf.argmax(y_test,1))    
                
                    ua = recall_score(fg,ff,average=None)
                    ua = np.sum(ua)/mood
                    UA.append(ua)
                                                                      
                    print("epoch %d, WA %g, UA %g" %(epoch, round(WA*100,2), round(UA*100,2)))

            if epoch == epoch_times:
                print('over')
                break

    return WA, UA

if __name__=='__main__':
    Model()





