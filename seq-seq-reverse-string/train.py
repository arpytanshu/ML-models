#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 10:04:39 2019

@author: ansh
"""

import time
import pickle
import random
import logging
import numpy as np
import tensorflow as tf

from os import system
from keras import Model
from keras import layers



#%%
'''
setup logging
generate sequence data
'''


DATA_DIR = './data/'
DATA_TYPE = 'illustrative_reverse' # could be 'copy' or 'reverse'
DATA_SIZE = 10000
MAXLEN_SEQ = 15
MINLEN_SEQ = 5
log_file = './objects/log.txt'



# create directories
system("if [ ! -d ./data ]; then echo 'Creating ./data'; mkdir ./data; fi")
system("if [ ! -d ./objects ]; then echo 'Creating ./objects'; mkdir ./objects; fi")



# setup logging
logging.basicConfig(filename = log_file, filemode='a', level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info('Logging Started')

def log_(message, end=None):
    print(message, end=end)
    logging.info(message)



# generate data using script from google/seq2seq/bin/tools/generate_toy_data.py 
generate_data = "python3 ./generate_toy_data.py --type {} --num_examples {} \
    --max_len {} --min_len {} --output_dir {}".format(DATA_TYPE, DATA_SIZE,
    MAXLEN_SEQ, MINLEN_SEQ, DATA_DIR)
   
log_("Generating data using {}".format(generate_data))
system(generate_data)




#%%
'''
read in the source and target sequences
'''
source_texts = []
target_texts = []
source_characters = set()
target_characters = set()


with open(DATA_DIR+'sources.txt', 'r') as f:
    for line in f:
        l = line.split()
        l.insert(len(l), '\n')
        source_texts.append(l)
        source_characters.update(l)
        
with open(DATA_DIR+'targets.txt', 'r') as f:
    for line in f:
        l = line.split()
        l.insert(len(l), '\n')
        l.insert(0, '\t')
        target_texts.append(l)
        target_characters.update(l)

# token => index mapping for source text
source_index_token = dict(list(enumerate(source_characters)))
# token => index mapping for target text
target_index_token = dict(list(enumerate(target_characters)))
# index => token mapping for source text
source_token_index = dict([(tkn, ix) for ix,tkn in source_index_token.items()])
# index => token mapping for target text
target_token_index = dict([(tkn, ix) for ix,tkn in target_index_token.items()])

num_source_tokens = source_index_token.__len__()
num_target_tokens = target_index_token.__len__()
max_source_seq_len = max([len(x) for x in source_texts])
max_target_seq_len = max([len(x) for x in target_texts])


log_("Number of tokens in source text: {}".format(num_source_tokens))
log_("Number of tokens in target text: {}".format(num_target_tokens))
log_("Maximum sequence length in source text: {}".format(max_source_seq_len))
log_("Maximum sequence length in target text: {}".format(max_target_seq_len))




#%%
'''
PREPARE INPUT FOR MODEL :

 - create one-hotted matrix for source & target sequences

 - matrix created from source texts will be used by encoder during training

 - 2 matrices will be created from target texts
     - decoder_input_data  : [ input to decoder while training ]
     - decoder_target_data : [ act as true labels when training ]
                             [ do not have '\t at beginning' ]
'''

encoder_input_data = np.zeros((DATA_SIZE, max_source_seq_len, num_source_tokens), dtype=np.float32)
decoder_input_data = np.zeros((DATA_SIZE, max_target_seq_len, num_target_tokens), dtype=np.float32)
decoder_target_data = np.zeros((DATA_SIZE, max_target_seq_len, num_target_tokens), dtype=np.float32)

for i, src_txt in enumerate(source_texts):
    for t, token in enumerate(src_txt):
        encoder_input_data[i, t, source_token_index[token]] = 1
    # pad sequence with \n if sequence length < max_source_seq_len
    encoder_input_data[i,t+1:, source_token_index['\n']] = 1

for i, trg_txt in enumerate(target_texts):
    for t, token in enumerate(trg_txt):
        decoder_input_data[i, t, target_token_index[token]] = 1
        if t>0:
            decoder_target_data[i,t-1, target_token_index[token]] = 1
    decoder_input_data[i,t+1:, target_token_index['\n']] = 1
    decoder_target_data[i, t:, target_token_index['\n']] = 1
    

#%%
'''
Training Model
'''    

lstm_units = 128 # num dimensions for encoding space
batch_size = 64
epochs = 50


## encoder
encoder_input = layers.Input(shape=(None, num_source_tokens), name='encoder_input')
encoder_lstm = layers.LSTM(lstm_units, return_state=True, name='encoder_lstm')

encoder_lstm_output, encoder_state_h, encoder_state_c = encoder_lstm(encoder_input)
encoder_states = [encoder_state_h, encoder_state_c]

## decoder
decoder_input = layers.Input(shape=(None, num_target_tokens), name='decoder_input')
decoder_lstm = layers.LSTM(lstm_units, return_state=True, return_sequences=True, name='decoder_lstm')
decoder_dense = layers.Dense(num_target_tokens, activation='softmax', name='decoder_dense')

decoder_lstm_output, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
decoder_dense_output = decoder_dense(decoder_lstm_output)


## train model
train_model = Model([encoder_input, decoder_input], decoder_dense_output)
train_model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])


'''
from keras.utils import plot_model
plot_model(train_model, show_shapes=True, show_layer_names=True, to_file='model.png')
'''

#%%
'''
Inference Model

 - inf_encoder_model : encodes sequences and returnes internal state of encoder LSTM
 - inf_decoder_model : takes in the encoded sequence and start token (in decoder_input)
                       returns the prediction & its hidden state (for next iteration)
'''

inf_encoder_model = Model(encoder_input, encoder_states, name='inference_encoder')

decoder_state_input_h = layers.Input(shape=(lstm_units, ), name='inference_decoder_state_input_h')
decoder_state_input_c = layers.Input(shape=(lstm_units, ), name='inference_decoder_state_input_c')
decoder_state_input = [decoder_state_input_h, decoder_state_input_c]

decoder_lstm_output, decoder_state_h, decoder_state_c = decoder_lstm(decoder_input, initial_state=decoder_state_input)

decoder_state_output = [decoder_state_h, decoder_state_c]

decoder_dense_output = decoder_dense(decoder_lstm_output)

inf_decoder_model = Model([ decoder_input ]         + decoder_state_input,
                          [ decoder_dense_output ]  + decoder_state_output)


#%%

def inference(input_sequence):
    
    encoded_sequence = inf_encoder_model(input_sequence)
    
    input_decoder = np.zeros((1,1,num_target_tokens))
    input_decoder[0,0, target_token_index['\t']] = 1
    
    stop_condition = False
    decoded_sequence = []
    
    while not stop_condition:
        sampled_output, h, c = inf_decoder_model.predict([input_decoder] + encoded_sequence, steps=1)
        
        sampled_token_index = np.argmax(sampled_output[0, -1, :])
        sampled_token = target_index_token[sampled_token_index]
        decoded_sequence.append(sampled_token)
        
        if (sampled_token == '\n' or len(decoded_sequence) > max_target_seq_len):
            stop_condition = True
        
        input_decoder = np.zeros((1,1,num_target_tokens))
        input_decoder[0,0, sampled_token_index] = 1
        
        encoded_sequence = [h, c]
    
    return decoded_sequence

def sample(num, log_f = None):
    for i in range(num):    
        seq_index = random.randint(0,DATA_SIZE)
        input_text = source_texts[seq_index]
        input_seq = np.zeros((1,max_source_seq_len,num_source_tokens))
        for t, token in enumerate(input_text):
            input_seq[0, t, source_token_index[token]] = 1
        # pad sequence with \n if sequence length < max_source_seq_len
        input_seq[0,t+1:, source_token_index['\n']] = 1
        
        input_seq_tf = tf.convert_to_tensor(input_seq, np.float32)
        
        decoded_sequence = inference(input_seq_tf)
        
        log_('Input sequence:\t{}'.format(' '.join(input_text).strip()))
        log_('Dcded sequence:\t{}\n'.format(' '.join(decoded_sequence).strip()))
        
HISTORY = [] 
    
#%%
'''
Train model & sample
'''

log_("Train Model");log_("###########\n\n")
train_model.summary(print_fn=log_)
log_("Inference Encoder Model");log_("#######################\n\n")
inf_encoder_model.summary(print_fn=log_)
log_("Inference Decoder Model");log_("#######################\n\n")
inf_decoder_model.summary(print_fn=log_)


log_("Starting Training . . .\n\n")


total_time = 0
for i in range(epochs):
    s = time.time()
    history = train_model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                batch_size=batch_size,
                epochs=1, verbose=0)
    e = time.time()
    epoch_time = e - s
    total_time += epoch_time
    log_("Epoch: {} Accuracy: {} Loss: {} Time: {}s ".format( i+1,
        round(float(history.history['accuracy'][0]), 3), round(float(history.history['loss'][0]), 3), round(epoch_time, 3)))

    HISTORY.extend(history.history)
    if(i%5==0): sample(2)

log_("Total train time: {} seconds\n".format(total_time))


#%%

'''
SAVE MODEL
'''
    
log_('Saving models to ./objects/*_small.h5\n')

train_model.save('./objects/train_model_small.h5')
inf_encoder_model.save('./objects/inf_encoder_small.h5')
inf_decoder_model.save('./objects/inf_decoder_small.h5')

log_('Saving train history to ./objects/history.pickle\n')
with open('./objects/history_small.pickle', 'wb') as f:
  pickle.dump(history, f)

character_mapping = {
    'source_index_token': source_index_token,
    'target_index_token': target_index_token,    
    }
log_('Saving character mapping dictionaries to ./objects/char_map_small.pickle\n')
with open('./objects/char_map_small.pickle', 'wb') as f:
  pickle.dump(character_mapping, f)

log_('Finished.')

