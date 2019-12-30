### Dataset
* Dataset contains 64 x 64 x 3 channel RGB images of hand symbols.
* Each hand symbol represents digits from [0, 1, 2, 3, 4, 5]
* train samples : 1080
* test samples : 120

![Dataset sample](https://github.com/arpytanshu/ML-models/blob/master/tensorflow_hand_dataset/dataset-sample.png)

### Data Preprocessing
* Unroll labels into one-hot vectors
* Mean normalize train and test samples around 255.

### Neural Network Model
Multi Layer Perceptron
* input layer : 64 x 64 x 3 = 12288
* hidden layer 1 : 25 units , activation = relu
* hidden layer 2 : 12 units , activation = relu
* output layer : 6 units , activation = sigmoid

### HyperParameters for training
* Optimizer : tf.train.AdamOptimizer
* Learning Rate = 0.001
* Epoch : 1500
* Minibatch size : 1500
* Dropout : no dropout

![Training](https://github.com/arpytanshu/ML-models/blob/master/tensorflow_hand_dataset/image1.png)

### Metrics
* Train Accuracy : 93.5
* Test Accuracy : 82.5
