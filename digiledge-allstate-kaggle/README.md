## Allstate Claims Severity
[Allstate Claims Severity - kaggle](https://www.kaggle.com/c/allstate-claims-severity/)

### Data
Each row in this dataset represents an insurance claim. You must predict the value for the 'loss' column. Variables prefaced with 'cat' are categorical, while those prefaced with 'cont' are continuous.

### Approach

Since all continuous features are normalized, we need not bother about that.  
Categorical features have cardinality from as low as 2 to 100+.  
I started off with trying to see correlated features, and there were quite many.  

To reduce the number of features, we first tried to estimate MAE using only the categorical features and only the continuous features alone, one at a time.
And then did PCA on both of them seperately to see any change in CV score.  

Then I used the f_regression and mutual_information tests to score the features. The features that scored badly on both these tests were discarded. The remaining feautres were kept.

Then I used a MLP model with the following parameters.  
```
class NN:
    def __init__(self):
        self.in_shape = common_features_union.__len__()
        self.num_layers = 3
        self.nodes = [2048,1024, 1]
        self.activations = ['relu', 'relu', 'relu']
        self.dropouts = [0.25,0.15,0]
        self.loss = 'mean_squared_logarithmic_error'
        self.optimizer = 'rmsprop'
```
After training for ~ 80 epochs, I got a MAE of 1160.xx on the public test data on kaggle.   
