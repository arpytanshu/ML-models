#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 21:23:12 2020

@author: ansh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 22:46:00 2019

@author: ansh
"""
###############################################################################

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection      import train_test_split

from keras.utils                  import to_categorical
from keras.preprocessing.text     import Tokenizer
from keras.preprocessing.sequence import pad_sequences

###############################################################################

object_path = './objects/'
train_file_path = './data/train.csv'
embedding_file_path = '../../../DATASETS/glove.6B.100d.txt'


VAL_SPLIT = 0.2
MAX_VOCABULARY = 20000
MAX_SEQ_LENGTHS = 50
EMBEDDING_DIM = 100



###############################################################################
'''
READ
GET TEST <-> VAL
'''


data = pd.read_csv(train_file_path)

# drop columns not required for training
data.drop(columns=['id', 'qid1', 'qid2'], inplace=True)

# seperate Xs and Ys
X = data[['question1', 'question2']]
Y = data.is_duplicate

del(data)

# train - test split. DO THIS RIGHT BEFORE FEEDING INTO MODEL.
# X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=VAL_SPLIT)

###############################################################################
'''
TOKENIZE SENTENCES
SAVE TOKENIZER OBJECT FOR LATER USE < PICKLE IT { achaar bana de } >
GET word_index and index_word mappings from tokenizer object
'''


all_questions = [str(x) for x in list(X.question1.append(X.question2).values)]



tokenizer = Tokenizer(MAX_VOCABULARY, oov_token='<UNK>')
tokenizer.fit_on_texts(all_questions)
# saving
with open(object_path+'tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# loading
with open(object_path+'tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

word_index = tokenizer.word_index
index_word = tokenizer.index_word



###############################################################################
'''
Transform all questions to sequences
CONVERT LABELS TO CATEGORICAL

'''
X_question1_sequences = tokenizer.texts_to_sequences([str(x) for x in X.question1.values])
X_question2_sequences = tokenizer.texts_to_sequences([str(x) for x in X.question2.values])

X_question1_padded = pad_sequences(X_question1_sequences, MAX_SEQ_LENGTHS)
X_question2_padded = pad_sequences(X_question2_sequences, MAX_SEQ_LENGTHS)


Y_categorical = to_categorical(Y, 2)

###############################################################################
'''
SETTLE ON A VOCAB SIZE
SETTLE ON A SEQUENCE LENGTH
'''
print('TOTAL UNIQUE TOKENS: ', tokenizer.word_counts.__len__())
counts = [val for _,val in tokenizer.word_counts.items()]
plt.hist(counts, log=False, bins=10)
plt.title("Histogram of count of each unique token in corpus. {in LOG scale}")
plt.show()

# Its clear, we will settle down for 20K top words


lengths1 = [len(x) for x in X_question2_sequences]
lengths2 = [len(x) for x in X_question1_sequences]
plt.hist(lengths1, bins=500)
plt.title('Histogram of length of question1.')
plt.show()
plt.hist(lengths2, bins=500)
plt.title('Histogram of length of question2.')
plt.show()

# we will settle for sequence lengths of 50

###############################################################################
'''
GET IN THE EMBEDDING FILE
PICKLE THE EMBEDDING DICT
CREATE EMBEDDING MATRIX FOR SEQUENCES
PICKLE EMBEDDING MATRIX
'''


embedding_data = {} # to hold our embeddings

embed_data_f = open(embedding_file_path, 'r')
for line in embed_data_f:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    embedding_data[word] = vector
embed_data_f.close()

# pickle the embedding dictionary

# saving
with open(object_path+'embeddings.pickle', 'wb') as handle:
    pickle.dump(embedding_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# loading
with open(object_path+'embeddings.pickle', 'rb') as handle:
    embedding_data = pickle.load(handle)
    

embedding_matrix = np.zeros((MAX_VOCABULARY, EMBEDDING_DIM ), dtype='float32')

for word, ix in word_index.items():
    if(ix<MAX_VOCABULARY):
        vector = embedding_data.get(word)
        if vector is not None:
            embedding_matrix[ix] = vector
            
# save embedding_matrix on disk 
np.save(object_path+'embedding_matrix.npy', embedding_matrix)



# load embedding_matrix from disk
embedding_matrix = np.load(object_path+'embedding_matrix.npy')



###############################################################################
'''
CREATE DRAFT MODEL
'''

from keras.layers import Dense, Embedding, Dropout, Input, LSTM, concatenate
from keras.models import Model
from keras.optimizers import Adam, rmsprop
from keras.losses import categorical_crossentropy

LSTM_UNITS          = 256
DO_LSTM             = 0.3
DO_RECURR           = 0.3
DENSE_UNITS         = [512, 512, 2]
DENSE_ACTIVATIONS   = ['relu', 'relu', 'sigmoid']

LOSS_FN             = categorical_crossentropy
OPTIMIZER           = rmsprop()
EPOCHS              = 1
BATCH_SIZE          = 1024
VAL_SPLIT           = 0.25



EmbeddingLayer1 = Embedding(MAX_VOCABULARY,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQ_LENGTHS,
                            trainable=False, name='embedding_x1')
EmbeddingLayer2 = Embedding(MAX_VOCABULARY,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQ_LENGTHS,
                            trainable=False, name='embedding_x2')


input_x1 =  Input(shape=(MAX_SEQ_LENGTHS,), name='input_x1')
x1 =        EmbeddingLayer1(input_x1)
x1 =        LSTM(LSTM_UNITS, dropout = DO_LSTM, recurrent_dropout = DO_RECURR, name = 'lstm_x1')(x1)

input_x2 =  Input(shape=(MAX_SEQ_LENGTHS,), name='input_x2')
x2 =        EmbeddingLayer2(input_x2)
x2 =        LSTM(LSTM_UNITS, dropout = DO_LSTM, recurrent_dropout = DO_RECURR, name = 'lstm_x2')(x2)

x =         concatenate([x1, x2], name='concatenate_x1_x2')
dense_1 =   Dense(DENSE_UNITS[0], activation = DENSE_ACTIVATIONS[0], name='dense_1')(x)
# dense_2 =   Dense(DENSE_UNITS[1], activation = DENSE_ACTIVATIONS[1], name='dense_2')(dense_1)
out     =   Dense(DENSE_UNITS[1], activation = DENSE_ACTIVATIONS[1], name='dense_out')(dense_1)

model = Model(inputs = [input_x1, input_x2], outputs = out)

model.compile(OPTIMIZER, LOSS_FN, metrics = ['accuracy'])

# model.summary()
# from keras.utils import plot_model
# plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True, to_file='model.png')


history = model.fit({'input_x1' : X_question1_padded, 'input_x2' : X_question2_padded},
                    {'dense_out' : Y_categorical},
                    epochs = EPOCHS,
                    batch_size = BATCH_SIZE,
                    validation_split = VAL_SPLIT
                    )


#%%

from keras.models import load_model
model = load_model(object_path+"model.h5")




submission = pd.read_csv('./data/sample_submission.csv')


test_file_path = './data/test.csv'


test_data = pd.read_csv(test_file_path)

test_question1 = tokenizer.texts_to_sequences([str(x) for x in test_data.question1.values])
test_question2 = tokenizer.texts_to_sequences([str(x) for x in test_data.question2.values])

test_question1_padded = pad_sequences(test_question1, MAX_SEQ_LENGTHS)
test_question2_padded = pad_sequences(test_question2, MAX_SEQ_LENGTHS)


np.save(object_path+'test_q1_padded.npy', test_question1_padded)
np.save(object_path+'test_q2_padded.npy', test_question2_padded)

# test_predictions = model.predict({'input_x1' : test_question1_padded, 'input_x2' : test_question2_padded})



test_question1_padded = np.load(object_path+'test_q1_padded.npy')
test_question2_padded = np.load(object_path+'test_q2_padded.npy')



#%%


import time

l = submission.__len__()
batch = 10000

test_predictions = []

for i in range(0,int(l/batch)+1):
    if(i != int(l/batch) ):
        start = (i*batch)
        end = (i*batch)+batch   
    else:
        start = int(l/batch)*batch
        end=-1
    print("inference batch => {} : {}".format(start, end))
    s = time.time()
    batch_preds =  model.predict({'input_x1' : test_question1_padded[start:end,:], 'input_x2' : test_question2_padded[start:end,:]}, batch_size=2048, verbose=True)
    e = time.time()    
    test_predictions.extend([0 if x[0]>x[1] else 1 for x in batch_preds])        
    print("Time elapsed: {} seconds.".format(e-s))    
        
        

submission['is_duplicate'] = test_predictions
submission.to_csv('submission_1.csv', index=False)

