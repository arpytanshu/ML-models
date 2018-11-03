#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 11:16:53 2018

@author: arpytanshu
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

train_labels = to_categorical(train_data.Survived, 2) #ready to export
#[0 : dead       : [1,0]]
#[1 : survived   : [0,1]]


#perform same operation on both train & test data
#concatenating train & test data together
train_data_count = train_data.PassengerId.count()
test_data_count = test_data.PassengerId.count()

data = pd.concat([train_data, test_data])


#this is a testing variable to choose if you want to repalace the features
#with linear values or values weighted a/c to survival %age
use_weighted_features = True



def show_pie_chart(col_name):
    print('stats for ' + col_name)
    classes = data[col_name].unique()
    for cls in classes:
        try:
            data.Survived[data[col_name] == cls ].value_counts().plot(kind='pie')
            plt.title('Class : {}'.format(cls))
            plt.show()
        except:
            print('something went wrong, but dont worry')


def show_stats(col_name):
    print('stats for ' + col_name)
    classes = data[col_name].unique()
    print(classes)
    
    for cls in classes:
        stat = data.Survived[data[col_name] == cls ].value_counts()
        print(cls)
        dead =0; alive=0;
        try:
            dead = stat[0.0]
        except:
            print('Nody in this class died!')
            
        try:
            alive = stat[1.0]
        except:
            print('Nody in this class lived!')
        
        if(dead != 0 or alive != 0):
            survival = alive / (dead + alive) * 100
            print('survival: ' + str(survival))
        else:
            print('cannot evaluate survival %age.')
        

        
#=================
## working with sex
#=================
replacement_sex = {'male': 1, 'female':0}
data['Sex'] = data['Sex'].apply(lambda x: replacement_sex.get(x))

#strip the crap out of the name, keep only titles
data['Name'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
titles = data.Name.unique()

#=================
##working with Age
#=================
#get median age of all titles,
#this data wil be used to fill in the missing fields in age column
data.Age.fillna(-1, inplace=True)
medians = {}
for title in titles:
    median = data.Age[data['Name'] == title].median()
    medians[title] = median
    data.Age[(data['Name']==title) & (data['Age'] == -1)] = medians[title]

#=================
##working with names
#=================
#stats for Name
#['Mr' 'Mrs' 'Miss' 'Master' 'Don' 'Rev' 'Dr' 'Mme' 'Ms' 'Major' 'Lady'
# 'Sir' 'Mlle' 'Col' 'Capt' 'the Countess' 'Jonkheer' 'Dona']
#							%age			weight	linear
#Mr		: 	survival: 15.667311411992262	1		1
#Mrs		:	survival: 79.2				8		6
#Miss	:	survival: 69.78021978021978		7		5
#Master	:	survival: 57.49999999999999		5		4
#Don		:	survival: 0.0					0		0
#Rev		:	survival: 0.0					0		0
#Dr		:	survival: 42.857142857142854	4		2
#Mme		:	survival: 100.0				10		7
#Ms		:	survival: 100.0				10		7
#Major	:	survival: 50.0				5		3
#Lady	:	survival: 100.0				10		7
#Sir		:	survival: 100.0				10		7
#Mlle	:	survival: 100.0				10		7
#Col		:	survival: 50.0				5		3
#Capt	:	survival: 0.0					0		0
#Countess:	survival: 100.0				10		7
#Jonkheer:	survival: 0.0					0		1
#Dona	:	only present in test data		-		-
replacement_name_weighted = { #new_modified
        'Mr'    :   1.5,
        'Mrs'   :   7.9,
        'Miss'  :   6.9,
        'Master':   5.7,
        'Don'   :   0,
        'Rev'   :   0,
        'Dr'    :   4.2,
        'Mme'   :   10,
        'Ms'    :   10,
        'Major' :   5,
        'Lady'  :   10,
        'Sir'   :   10,
        'Mlle'  :   10,
        'Col'   :   5,
        'Capt'  :   0,
        'the Countess'  :   10,
        'Jonkheer'  :   0,
        'Dona'  :   4
        }
replacement_name_linear = {
        'Mr'    :   1,
        'Mrs'   :   6,
        'Miss'  :   5,
        'Master':   4,
        'Don'   :   0,
        'Rev'   :   0,
        'Dr'    :   2,
        'Mme'   :   7,
        'Ms'    :   7,
        'Major' :   3,
        'Lady'  :   7,
        'Sir'   :   7,
        'Mlle'  :   7,
        'Col'   :   3,
        'Capt'  :   0,
        'the Countess'  :   7,
        'Jonkheer'  :   1,
        'Dona'  :   6
        }
    
#=================
#working with embarked
#=================
#stats for Embarked
#['S' 'C' 'Q' nan]  %age            weight  linear  
#S : survival: 33.69565217391305    3      0
#C : survival: 55.35714285714286    5      2
#Q : survival: 38.961038961038966   4      1
replacement_embarked_weighted = {
        'C': 5.5,
        'Q': 4.8,
        'S': 3.3
        }
replacement_embarked_linear = { # linear
        'C': 2,
        'Q': 1,
        'S': 0
        }


#=================
#working with Parch
#=================
#stats for Parch
#[0 1 2 5 3 4 6 9]
#                    %age            weight  linear
#0   survival: 34.365781710914455    3       2  
#1   survival: 55.08474576271186     5       4
#2   survival: 50.0                  5       3
#3   survival: 60.0                  6       5
#4   survival: 0.0                   0       0
#5   survival: 20.0                  2       1
#6   survival: 0.0                   0       0
#9   only in test data               0       0
replacement_parch_weighted = {
        0: 3.4,
        1: 5.5,
        2: 5,
        3: 6,
        4: 0,
        5: 2,
        6: 0,
        9: 0
        }
replacement_parch_linear = {
        0: 2,
        1: 4,
        2: 3,
        3: 5,
        4: 0,
        5: 1,
        6: 0,
        9: 0
        }


#=================
#working with SibSp
#=================
#stats for SibSp
#[1 0 3 4 2 5 8]
#                        weighted  linear
#0   survival: 34.5394       3       3
#1   survival: 53.5885       5       5
#2   survival: 46.42857      4       4
#3   survival: 25.0          2       2
#4   survival: 16.6666       1       1
#5   survival: 0.0           0       0
#8   survival: 0.0           0       0
replacement_sibsp_weighted = {
        0: 3.4,
        1: 5.3,
        2: 4.6,
        3: 2.5,
        4: 1.6,
        5: 0,
        8: 0
        }
replacement_sibsp_linear = {
        0: 3,
        1: 5,
        2: 4,
        3: 2,
        4: 1,
        5: 0,
        8: 0
        }


#=================
#working with cabin
#=================
data.Cabin.fillna('U', inplace=True)
data['Cabin'] = data.Cabin.apply(lambda x: x[0])

#stats for Cabin
#['U' 'C' 'E' 'G' 'D' 'A' 'B' 'F' 'T']
#                 %age   weighted    linear
#U   survival: 29.9854      3        1
#C   survival: 59.3220      6        4
#E   survival: 75.0         7        5
#G   survival: 50.0         5        3
#D   survival: 75.7575      7        5
#A   survival: 46.6666      4        2
#B   survival: 74.4680      7        5
#F   survival: 61.5384      6        3
#T   survival: 0.0          0        0

replacement_cabin_weighted = {
        'U':2.9,
        'C':5.9,
        'E':7.5,
        'G':5,
        'D':7.5,
        'A':4.6,
        'B':7.4,
        'F':6.1,
        'T':0   
        }
replacement_cabin_linear = {
        'U':1,
        'C':4,
        'E':5,
        'G':3,
        'D':5,
        'A':2,
        'B':5,
        'F':3,
        'T':0   
        }
    
#=================
#drop insignificant features and labels
#=================

data.drop('Ticket', axis=1, inplace=True)
data.drop('Survived', axis=1, inplace=True)


#=================
#perforn feature replacement
#=================
if (use_weighted_features):
    data['Name'] = data['Name'].apply(lambda x: replacement_name_weighted.get(x))
    data['Cabin'] = data.Cabin.apply(lambda x: replacement_cabin_weighted.get(x))
    data['SibSp'] = data['SibSp'].apply(lambda x: replacement_sibsp_weighted.get(x))
    data['Parch'] = data['Parch'].apply(lambda x: replacement_parch_weighted.get(x))
    data['Embarked'] = data['Embarked'].apply(lambda x: replacement_embarked_weighted.get(x))
else:
    data['Name'] = data['Name'].apply(lambda x: replacement_name_linear.get(x))
    data['Cabin'] = data.Cabin.apply(lambda x: replacement_cabin_linear.get(x))
    data['SibSp'] = data['SibSp'].apply(lambda x: replacement_sibsp_linear.get(x))
    data['Parch'] = data['Parch'].apply(lambda x: replacement_parch_linear.get(x))
    data['Embarked'] = data['Embarked'].apply(lambda x: replacement_embarked_linear.get(x))  


#remove nan values so scaling could be applied
data['Fare'] = data['Fare'].fillna(data.Fare.median())
data['Embarked'] = data['Embarked'].fillna(data.Embarked.median())



#scale all data (except sex)
from sklearn.preprocessing import StandardScaler

data['Name'] = StandardScaler().fit_transform(data['Name'].values.reshape(-1, 1))
data['Fare'] = StandardScaler().fit_transform(data['Fare'].values.reshape(-1, 1))
data['Embarked'] = StandardScaler().fit_transform(data['Embarked'].values.reshape(-1, 1))
data['Cabin'] = StandardScaler().fit_transform(data['Cabin'].values.reshape(-1, 1))
data['Parch'] = StandardScaler().fit_transform(data['Parch'].values.reshape(-1, 1))
data['SibSp'] = StandardScaler().fit_transform(data['SibSp'].values.reshape(-1, 1))
data['Age'] = StandardScaler().fit_transform(data['Age'].values.reshape(-1, 1))
data['Pclass'] = StandardScaler().fit_transform(data['Pclass'].values.reshape(-1, 1))



features = data.keys()
train_df = data[0:train_data_count][:]
test_df = data[train_data_count:][:]

data.drop('PassengerId', axis=1, inplace=True)


train_x = data[0:train_data_count][:].values
test_x = data[train_data_count:][:].values

#[0 : dead       : [1,0]]
#[1 : survived   : [0,1]]
train_y =  train_labels     #ready to export




##################################################################################


nn_arch = [9,36,36,2]


model =  keras.Sequential([
            keras.layers.Dense(nn_arch[1], activation=tf.nn.sigmoid, input_shape=(nn_arch[0],), kernel_initializer=keras.initializers.glorot_normal(),
                               bias_initializer=keras.initializers.zeros()),
            keras.layers.Dropout(0.015),
            keras.layers.Dense(nn_arch[2], activation=tf.nn.sigmoid, kernel_initializer=keras.initializers.glorot_normal(),
                               bias_initializer=keras.initializers.zeros()),
            
            keras.layers.Dropout(0.015),
            keras.layers.Dense(nn_arch[3], activation='softmax', kernel_initializer=keras.initializers.glorot_normal(),
                               bias_initializer=keras.initializers.zeros())]
      )

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=tf.train.AdamOptimizer(0.025) , metrics=['accuracy'])


history = model.fit(train_x, train_y, epochs = 300, batch_size=32, verbose=0)

predictions_prob = model.predict(test_x)
predictions = np.argmax(predictions_prob, axis=1)



submission = pd.DataFrame({'PassengerId':test_df['PassengerId'], 'Survived':predictions})
filename = "titanic_preds_accu-" + str(history.history['acc'][-1])[2:4] + "_loss-" + str(history.history['loss'][-1])[2:4] + ".csv"
submission.to_csv(filename, index=False)
print('Saved file: '+ filename)




###########################
#plot losses
###########################

training_loss = history.history['loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

