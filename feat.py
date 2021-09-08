
import numpy as np
import os
import cv2  
from itertools import chain
import librosa

def feature():
    
    filepath = "G:/Experiment/Database/EMO-DB/all/" 
    length = 800

    lis_1 = []
    lis_2 = []
    n = []
    winlen = 400
    winstep = 160
    
    filename= os.listdir(filepath)
    for file in filename:
        
        y, sr = librosa.load(filepath+file, sr=None)    
           
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=winlen,
                                                 hop_length=winstep,n_mels=64, fmin=0.0, fmax=8000)
        processed_audio = librosa.power_to_db(melspec)

        delta1 = librosa.feature.delta(processed_audio, width=3)   
        delta2 = librosa.feature.delta(processed_audio, order=2, width=3)
        
        col1 =  np.transpose(processed_audio)
        col2 =  np.transpose(delta1)
        col3 =  np.transpose(delta2)
        
        col1=cv2.resize(col1,                        
                        (int(227), int(227)),
                        interpolation=cv2.INTER_LINEAR)
        col2=cv2.resize(col2,
                        (int(227), int(227)),
                        interpolation=cv2.INTER_LINEAR)
        col3=cv2.resize(col3,
                        (int(227), int(227)),
                       interpolation=cv2.INTER_LINEAR)
        su=list(chain.from_iterable(zip(col1, col2, col3)))
        lis_1.append(su)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15, n_fft=winlen, hop_length=winstep, fmin=0.0, fmax=8000)      
        delta1_mfcc = librosa.feature.delta(mfcc, width=3)      
    
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=25, n_fft=winlen, hop_length=winstep, fmin=0.0, fmax=8000)
        mel = librosa.power_to_db(mel)               
        delta1_mel = librosa.feature.delta(mel, width=3)      
     
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=winlen, hop_length=winstep,)        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=winlen, hop_length=winstep,)       
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length = winlen, hop_length = winstep)             
        rms = librosa.feature.rms(y=y, frame_length=winlen, hop_length=winstep)  

        fe = np.vstack((mfcc, delta1_mfcc, mel, delta1_mel,  flatness, chroma, zcr, rms)) 
        
        col1 = np.transpose(fe)         
       
        ad = length - len(col1)
        n.append(len(col1))
        col1 = np.row_stack((col1,np.zeros([ad, len(fe)])))
    
        lis_2.append(col1)                      
            
    x_1=np.reshape(lis_1, [-1, 227, 227, 3])      
    x_2=np.reshape(lis_2, [-1, length, len(fe)])      
    
    return  x_1, x_2, n,
        



