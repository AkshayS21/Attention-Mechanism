from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt

repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') 
dotor = Dot(axes = 1)


def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """
    
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) to
    #concatenate it with all hidden states "a" 
    s_prev = repeator(s_prev)
   # print(s_prev.shape)
    # Use concatenator to concatenate a and s_prev on the last axis 
    concat = concatenator([a , s_prev])
    #print(concat.shape)
    
    # Use densor1 to propagate concat through a small fully-connected 
    # neural network to compute the "intermediate energies" variable e. 
    e = densor1(concat)
    #print(e.shape)
    # Use densor2 to propagate e through a small fully-connected neural 
    # network to compute the "energies" variable energies. 
    energies = densor2(e)
    #print(energies.shape)
    # Use "activator" on "energies" to compute the attention weights "alphas" 
    alphas = activator(energies)
    #print(alphas.shape)
    # Use dotor together with "alphas" and "a" to compute the context vector
    # to be given to the next (post-attention) LSTM-cell 
    context = dotor([alphas,a])
    #print(context.shape)
    #print(context.shape)
    
    return context