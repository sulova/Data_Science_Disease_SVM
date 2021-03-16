# Data Science - Support Vector Machine (SVM) Multiclass Classification

In this project, I will be talking about a Machine Learning Model called Support Vector Machine (SVM). It is a very powerful and versatile supervised Machine Learning model, superivsed meaning sample data should be labeled. Algorithm works perfectly for almost any supervised problem and it can be used for **classification** or **regression problems**. There is a slight difference in the implementation of these algorithms, *Support Vector Classifier* and *Support Vector Regressor*.

# Support Vector Machine (SVM)
SVM is well suited for classification of complex but small or medium sized datasets. To generalize, the objective is to find a hyperplane that maximizes the separation of the data points to their potential classes in an n-dimensional space. The data points with the minimum distance to the hyperplane  are called Support Vectors.

- **The One-to-Rest approach** - separate between every two classes. Each SVM would predict membership in one of the **m** classes. This means the separation takes into account only the points of the two classes in the current split. Thus, the red-blue line tries to maximize the separation only between blue and red points and It has nothing to do with green points.

- **The One-to-One approach** - separate between a class and all others at once, meaning the separation takes all points into account, dividing them into two groups; a group for the class points and a group for all other points. Thus, the green line tries to maximize the separation between green points and all other points at once.

<div align="center">
  Example of 3 classes classification
</div>

<p align="center">
  <img width="600" height="300" src="https://github.com/sulova/Data_Science_Disease_SVM/blob/main/SVM.PNG ">
</p>

**The advantages of SVM**
 - Effective in high dimensional spaces
 - Effective in cases where number of dimensions is greater than the number of samples
 - Memory efficient as it uses a subset of training points in the decision function (called support vectors)
 
Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

**The disadvantages of SVM include**
 - If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial
 - Do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation
 - Do not give the best performance for handling text structures as compared to other algorithms that are used in handling text data. 

Training Support Vector Machines for Multiclass Classification

```python
import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt 
%matplotlib inline
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
```

Any results you write to the current directory are saved as output.

```python
import os
print(os.listdir("../input"))
```

Loading both test and train set.

```python
train = shuffle(pd.read_csv("../input/train.csv"))
test = shuffle(pd.read_csv("../input/test.csv"))
```

Check for missing values in the dataset
```python
print("Any missing sample in training set:",train.isnull().values.any())
print("Any missing sample in test set:",test.isnull().values.any(), "\n")
```
Visualizing Outcome Distribution
```python
temp = train["Class_column"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
labels = df['labels']
sizes = df['values']
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','cyan','lightpink']
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90, pctdistance=1.1, labeldistance=1.2)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()
```
Seperating Predictors and Outcome values from train and test sets
```python
X_train = pd.DataFrame(train.drop(['Activity','subject'],axis=1))
Y_train_label = train.Activity.values.astype(object)
X_test = pd.DataFrame(test.drop(['Activity','subject'],axis=1))
Y_test_label = test.Activity.values.astype(object)

# Dimension of Train and Test set 
print("Dimension of Train set",X_train.shape)
print("Dimension of Test set",X_test.shape,"\n")
```

