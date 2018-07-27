#### DATASET
Fashion MNIST
train samples : 60000
test samples : 10000

#### Model
tf keras
Dense(200) -> Dropout -> Dense(200) -> Dropout -> Dense(10)

Layer (type)                 Output Shape              Param
-------------------------------------------------------------
flatten (Flatten)            (None, 784)               0         
dense (Dense)                (None, 200)               157000    
dropout (Dropout)            (None, 200)               0         
dense_1 (Dense)              (None, 200)               40200     
dropout_1 (Dropout)          (None, 200)               0         
dense_2 (Dense)              (None, 10)                2010      

#### Metrics
Test Accuracy : 88%
