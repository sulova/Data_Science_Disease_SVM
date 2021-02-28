# Data Science Disease - SVM 

In this project, I will be talking about a Machine Learning Model called Support Vector Machine (SVM). It is a very powerful and versatile supervised Machine Learning model,meaning sample data should be labeled. lgorithm works perfectly for almost any supervised problem and it can be used for **classification** or **regression problems**.  

# Support Vector Machine (SVM)
SVM is well suited for classification of complex but small or medium sized datasets. To generalize, the objective is to find a hyperplane that maximizes the separation of the data points to their potential classes in an n-dimensional space. The data points with the minimum distance to the hyperplane  are called Support Vectors.

In the One-to-Rest approach, the classifier can use \pmb{m} SVMs. Each SVM would predict membership in one of the \pmb{m} classes.
In the One-to-One approach, the classifier can use \pmb{\frac{m (m-1)}{2}} SVMs.

*Classify dataset from M classes data set*
![Image](https://github.com/sulova/Data_Science_Disease_SVM/blob/main/SVM.PNG)

The method which is used for classification is called *Support Vector Classifier* and the method which is used for regression is called *Support Vector Regressor*.There is a slight difference in the implementation of these algorithms.

*Support Vector Regressor*

*Support Vector Classifier*


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
