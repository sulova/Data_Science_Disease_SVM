# Data Science - Support Vector Machine (SVM) Multiclass Classification



In this project, I will be talking about a Machine Learning Model called Support Vector Machine (SVM). It is a very powerful and versatile supervised Machine Learning model, superivsed meaning sample data should be labeled. Algorithm works perfectly for almost any supervised problem and it can be used for **classification** or **regression problems**. There is a slight difference in the implementation of these algorithms, *Support Vector Classifier* and *Support Vector Regressor*.

# Support Vector Machine (SVM)
SVM is well suited for classification of complex but small or medium sized datasets. To generalize, the objective is to find a hyperplane that maximizes the separation of the data points to their potential classes in an n-dimensional space. The data points with the minimum distance to the hyperplane  are called Support Vectors.

- **The One-to-Rest approach** - separate between every two classes. Each SVM would predict membership in one of the **m** classes. This means the separation takes into account only the points of the two classes in the current split. Thus, the red-blue line tries to maximize the separation only between blue and red points and It has nothing to do with green points.

- **The One-to-One approach** - separate between a class and all others at once, meaning the separation takes all points into account, dividing them into two groups; a group for the class points and a group for all other points. Thus, the green line tries to maximize the separation between green points and all other points at once.

*Example of 3 classes classification*
![Image](https://github.com/sulova/Data_Science_Disease_SVM/blob/main/SVM.PNG)


**The advantages of SVM**
 - Effective in high dimensional spaces
 - Effective in cases where number of dimensions is greater than the number of samples
 - memory efficient as it uses a subset of training points in the decision function (called support vectors)
 
Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

**The disadvantages of SVM include**
 - If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial
 - Do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation
 - Do not give the best performance for handling text structures as compared to other algorithms that are used in handling text data. 

# Exploratory Data Analysis
Now we will make necessary imports and try to load the dataset to jupyter notebook.

```python
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv('/disease.data')
```
