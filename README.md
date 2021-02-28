# Data Science Disease - SVM 

Data analysis to predict disease using **Support Vector Machine (SVM)**. A data science enthusiast who works on ML, SVM algorithm works perfectly for almost any supervised problem.
It can be used for **classification** or **regression problems**.

**Prediction**
We can use a model to make predictions, or to estimate a dependent variable’s value given at least one independent variable’s value.

# Support Vector Machine (SVM)
In this project, we will be talking about a Machine Learning Model called Support Vector Machine (SVM). 
The method which is used for classification is called /Support Vector Classifier” and the method which is used for regression is called “Support Vector Regressor”.
A Support Vector Machine (SVM) is a binary linear classification whose decision boundary is explicitly constructed to minimize generalization error. It is a very powerful and versatile Machine Learning model, capable of performing linear or nonlinear classification, regression and even outlier detection.
SVM is well suited for classification of complex but small or medium sized datasets.

**The advantages of SVM**
 - Effective in high dimensional spaces
 - Effective in cases where number of dimensions is greater than the number of samples
 - memory efficient as it uses a subset of training points in the decision function (called support vectors)
 
Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

**The disadvantages of SVM include**
 - If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial
 - Do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation

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
