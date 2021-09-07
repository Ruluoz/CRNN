# CRNN-MA
''Convolutional-Recurrent Neural Networks with Multiple Attention Mechanisms for Speech Emotion Recognition''. We set three strategies for the proposed CRNN-MA: A multiple self-attention layer in the CNN module on frame-level weights, a multi-dimensional attention layer as the input features of the LSTM, and a fusion layer summarizing the features of the two modules. 

# Setup
tensorflow == 1.4  
python == 3.6

# Description
## feat.py
To extract speech features. filepath is the path where the database is located.    

## Module_attlstm and Module_cnn.py 
CNN module and LSTM module

##model.py
main program
