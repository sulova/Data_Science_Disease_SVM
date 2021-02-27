# Data Science Disease

Using Data Analytics to Predict disease Using Support Vector Machine (SVM).

Prediction
We can use a model to make predictions, or to estimate a dependent variable’s value given at least one independent variable’s value.

# Support Vector Machine (SVM)
In this project, we will be talking about a Machine Learning Model called Support Vector Machine (SVM).
A Support Vector Machine (SVM) is a binary linear classification whose decision boundary is explicitly constructed to minimize generalization error. It is a very powerful and versatile Machine Learning model, capable of performing linear or nonlinear classification, regression and even outlier detection.
SVM is well suited for classification of complex but small or medium sized datasets.

The advantages of SVM
 - Effective in high dimensional spaces
 - Effective in cases where number of dimensions is greater than the number of samples
 - memory efficient as it uses a subset of training points in the decision function (called support vectors)
 
Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of SVM include:
 - If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial
 - Do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation
