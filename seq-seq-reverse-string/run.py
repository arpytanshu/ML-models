#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 23:51:17 2019

@author: ansh
"""

import pickle
import numpy as np
from keras.models import load_model
from tensorflow import convert_to_tensor

inference_encoder_path = './objects/inf_encoder_big.h5'
inference_decoder_path = './objects/inf_decoder_big.h5'
character_map_path = './objects/char_map_big.pickle'

with open(character_map_path, 'rb') as f:
    char_map = pickle.load(f)

source_index_token = char_map['source_index_token']
target_index_token = char_map['target_index_token']
num_source_tokens = char_map['num_source_tokens']
num_target_tokens = char_map['num_target_tokens']
max_source_seq_len = char_map['max_source_seq_len']
max_target_seq_len = char_map['max_target_seq_len']


source_token_index = dict([(tkn, ix) for ix,tkn in source_index_token.items()])
# index => token mapping for target text
target_token_index = dict([(tkn, ix) for ix,tkn in target_index_token.items()])



inf_encoder_model = load_model(inference_encoder_path)
inf_decoder_model = load_model(inference_decoder_path)

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
        if (sampled_token == '\n' or (len(decoded_sequence) > max_target_seq_len)):
            stop_condition = True
        
        input_decoder = np.zeros((1,1,num_target_tokens))
        input_decoder[0,0, sampled_token_index] = 1
        
        encoded_sequence = [h, c]
    
    return decoded_sequence


def run_big(text):
    if(len(text) > max_target_seq_len - 2):
        print('Model was built for text length <= {}. Try a shorter string.'.format(max_target_seq_len - 2))
        return False
    input_text = text + '\n'
    input_seq = np.zeros((1,max_source_seq_len,num_source_tokens))
    for t, token in enumerate(input_text):
        input_seq[0, t, source_token_index[token]] = 1
    # pad sequence with \n if sequence length < max_source_seq_len
    input_seq[0,t+1:, source_token_index['\n']] = 1

    input_seq_tf = convert_to_tensor(input_seq, np.float32)

    decoded_sequence = inference(input_seq_tf)

    print('Input sequence:\t{}'.format(' '.join(input_text).strip()))
    print('Dcded sequence:\t{}\n'.format(' '.join(decoded_sequence).strip()))
    return True

