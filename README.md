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

# Data Wrangling 

Data manipulation is another way to refer to this process is data manipulation but there is no set list or order of operations. However, there are three common tasks involved in the data wrangling process such as:
  - **Data cleaning**
  - **Data transformation**
  - **Data enrichment**

Useful techniques used to clean and process the data is with **Pandas library**. Pandas is a powerful toolkit for analyzing initial data and introducing dataset structures in Python. Activating Pandas is very easy in python. As one library to do the initial process of data analysis. 
Let's explore data and its types.

- **The overview of the dataset**

```python 
import pandas as pd
import seaborn as sns

df = pd.read_csv('/disease.data')

# to look at a small sample of the dataset at the top
df.head()

# to look at a small sample of the dataset at the end
df.tail()

# have a look at a subset of the rows or columns fx: select the first 10 columns
df.iloc[:,:10].head()

# shows the data type for each column, among other things
df.info()

# shows the data type for each column
df.dtypes()

# describe() gives the insights about the data and some useful statistics about the data such as mean, min and max etc.
df.describe()
```
- **Missing and duplicate value**

The dataset may consist of a lot of missing and duplicate values, so let's deal with them before applying any machine learning algorithms on them. If you have identified the missing values in the dataset, now you have a couple of options to deal with them, either we can drop those rows which consist missing values or calculate the mean, min, max and median etc.
  
```python
# dealing with missing values
df.isna().sum()

# Visualize the missing values using Missingno library. 
msno.matrix(df)
```

<div align="center">
 Visualize the missing values
</div>

<p align="center">
  <img width="400" height="200" src="https://github.com/sulova/Data_Science_Disease_SVM/blob/main/Sk%C3%A6rmbillede%202021-03-04%20212910.png ">
</p>

```python

# fill in the missing values in 'ColumnName'
ColumnName_mean_value = df['ColumnName'].mean()
df['ColumnName'] = df['ColumnName'].fillna(ColumnName_mean_value)

Replace the NaN with median.

# Remove 'ColumnName' column
df.drop("ColumnName",axis = 1,inplace=True)
# Drops all columns in the DataFrame that have more than 10% Null values
df.dropna(1,thresh=len(df.index)*0.9,inplace=True)

# List down all the duplicated rows in the dataframe
df[df.duplicated()]
# Remove those rows 
df.drop_duplicates(inplace=False) 

# Get rid of all non-unique columns in a dataset
nunique = df.apply(pd.Series.nunique)
cols_to_drop = nunique[nunique == 1].index
df.drop(cols_to_drop, axis=1,inplace=True)

# Check if any existing column has all null values
df.isnull().values.all(axis=0)
# Replace all the null values in the column with a zero
df['ColumnName'].fillna(0, inplace = True)

# Define in which columns to look for missing values
df.dropna(subset=['ColumnName_1', 'ColumnName_1'])
# Drop the rows where all elements are missing
df.dropna(how='all')
# Keep only the rows with at least 2 non-NA values
df.dropna(thresh=2)
```

- **Correct Data type for a column**

```python
# Conversion between data types for a column. 
 df['ColumnName'].astype()

# Change the labeling for better visualization and interpretation.**
df['target'] = df.target.replace({1: "Disease", 0: "No_disease"})
df['sex'] = df.sex.replace({1: "Male", 0: "Female"})

``` 

- **Filtering Data**

 ```python
# The value of the Country column is not USA
df[df[‘Country’]!=’USA’]

# The following piece of code filters the entire dataset for age greater than 40
filtered_age = df[df.Age>40]
filtered_age

# Grouping/Aggregating the values of the Customer column based on the Country column
df.groupby([‘Country’]).CustomerNo.count()

#  Merging the Customer with the Location DataFrame using a left join
pd.merge(customer,location,how=’left’,on=[‘city’])

``` 
A good data wrangler knows how to integrate information from multiple data sources, solving common transformation problems, and resolve data cleansing and quality issues.

# Exploratory Data Analysis
Exploratory Data Analysis (EDA) is a pre-processing step to understand the data. There are numerous methods and steps in performing EDA, however, most of them are specific, focusing on either visualization or distribution, and are incomplete.
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

print(df['SalePrice'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(df['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4});

```


Now lets take a look at how the housing price is distributed
