# Machine-learning-

# machine learning

Machine learning is the science of getting computers to act without being explicitly programmed. In the past decade, machine learning has given us self-driving cars, practical speech recognition, effective web search, and a vastly improved understanding of the human genome.

Machine Learning is a first-class ticket to the most exciting careers in data science. As data sources proliferate along with the computing power to process them, automated predictions have become much more accurate and dependable. Machine learning brings together computer science and statistics to harness that predictive power. It’s a must-have skill for all aspiring data analysts and data scientists, or anyone else who wants to wrestle all that raw data into refined trends and predictions.

In this modules, broadly I will talk about supervised as well as unsupervised learning. We will talk about multiple types of classifiers like Naïve Bayes, KNN, decision trees, SVMs, artificial neural networks, logistic regression, and ensemble learning. Further, we will also talk about linear regression analysis, sequence labeling using HMMs. As part of unsupervised learning, I will discuss clustering as well as dimensionality reduction. Finally, we will also discuss briefly semi-supervised learning, multi-task learning, architecting ML solutions, and a few ML case studies.


![image](https://user-images.githubusercontent.com/67232573/113742925-f5166800-96b7-11eb-9d98-c09c010752f0.png)




# Introduction to Machine Learning
Introduction to machine learning

Supervised, semisupervised, unsupervised machine learning

Types of data sets

Data() in R

Introduction to classification


![image](https://user-images.githubusercontent.com/67232573/113742996-052e4780-96b8-11eb-96be-7d37c52889c0.png)




# Decision Trees

Introduction to Decision tree

Hunt’s algorithm for learning a decision tree

Details of tree induction

GINI index computation

ID3, Entropy and information gain

ID3 Example

C4.5

Pruning

Metrics for performance Evaluation

Iris Decision Tree Example



![image](https://user-images.githubusercontent.com/67232573/113743225-46265c00-96b8-11eb-8da9-d86ae28e4c8d.png)




# K Nearest Neighbors (KNN)

Introduction to KNN algorithm

Decision boundary KNN Vs Decision tree

What is the best K

KNN Problem

Feature selection using KNNs

Wilson Editing

KNN Imputation

Speeding up KNN using KMeans

Coding up KNN from scratch in Python

KNN using sklearn

Digits classification using KNN in Python


![image](https://user-images.githubusercontent.com/67232573/113743348-6f46ec80-96b8-11eb-870c-8d9a169cb212.png)




# Naïve Bayes

Examples of few text classification problems

Classification for text using bag of words

Naïve Bayes for text classification

Multinomial Naïve Bayes

Multinomial Naïve Bayes Example

Naïve Bayes for Hand-written digit recognition

Naïve Bayes for weather data

Numeric stability issue with Naïve bayes

Gaussian Naïve Bayes from scratch in Python

Naïve Bayes using sklearn

Multinomial Naïve Bayes


![image](https://user-images.githubusercontent.com/67232573/113743511-9c939a80-96b8-11eb-87de-934b8a4a97c5.png)


![image](https://user-images.githubusercontent.com/67232573/113743593-b1702e00-96b8-11eb-8394-d7defc1dea39.png)



# SVMs

Linear Classifiers

Margin of SVM’s

SVM optimization

SVM for Data which is not linear separable

Learning non-linear patterns

Kernel Trick

SVM Parameter Tuning

Handling class imbalance in SVM’s

SVM’s pros and cons and summary

Linear SVM using Python

SVM with RBF kernel with Python

Learning SVM with noise data in Python



![image](https://user-images.githubusercontent.com/67232573/113743678-c9e04880-96b8-11eb-9ab1-099cc4212a51.png)


![image](https://user-images.githubusercontent.com/67232573/113743698-d06ec000-96b8-11eb-917b-7c60adf11672.png)




# Ensemble Learning

Introduction to Ensemble learning

Why Ensemble learning

Independently constructed ensembles for classification: Majority voting

Independently constructed ensembles for classification: Bagging

Independently constructed ensembles for classification: Random forests

Independently constructed ensembles for classification: Error correcting output codes

Sequentially constructed ensembles for classification boosting

Sequentially constructed ensembles for classification boosting example

Sequentially constructed ensembles for classification stacking

Introduction to gradient boosted machines (GBM)

Relations between GBM gradient Descent

GBM regression with squared loss

Bagging in Python

Random forests in Python

Boosting in Python

Feature importance using ensemble classifiers

XGBoost in Python

Parameter tuning for GBM’s

Voting classifier using skLearn


![image](https://user-images.githubusercontent.com/67232573/113743832-f3996f80-96b8-11eb-9341-83c0a629516e.png)




# Artificial Neural Networks

Motivation for Artificial Neural Network

Mimicing a single neuron, integration function, Activation Function

Perceptron Algorithm

Perceptron Algorithm Example

Decision Boundary for a single Neuron

Learning Non-Linear Patterns

Introduction to Deep Learning

What can we achieve using a single hidden layers

MLPs with Sigmoid activation Function

Layers are transformation into a new space

Playing at the Tensorflow playground

Cost function, Loss function, Error Surface

How to learn Weights

Stochastic Gradient descent, Minibatch SGD, Momentum

Choosing a learning Rate

Updaters

Back Propagation

Softmax and Binary/Multi-class cross entropy loss

Overfitting and Regularization

Practical Advice on using Neural Networks

Autonomous Vehicles

Automated Feature Learning using Neural Networks

Deep Learning Architectures and Libraries

Applications of Artificial Neural Networks

History of Artificial Neural Networks and Revival

Python Code: Basic Introduction to Tensorflow: Constants, Placeholders and Variables.

Python Code: Learning the first Tensorflow model: Linear Regression using Tensorflow.

Python Code: MLP for Hand-written digit recognition with no hidden layer with 10 output neurons

Python Code: MLP for Hand-written digit recognition with two hidden layers

Python Code: Fashion Multi-class classification using MLP in Keras





# Linear Regression

Introduction to Linear regression

Understanding the real meaning of Linear Regression

𝑹^𝟐: Coefficient of Determination

Multiple Linear Regression and Non-linear Regression

Assumptions for Linear Regression

Using Residual to Verify the Assumptions for Linear Regression

Deriving Linear Regression Formulas using Ordinary Least Squares Method

Multiple Linear Regression

Underfitting, Overfitting, Bias and Variance

Ridge Regularization

Lasso Regularization, Elastic Net Regularization

Metrics and Practical Considerations for Regression

Python code: Simple Linear Regression using sklearn

Python code: Example to code up regression using ordinary least squares method

Python code: Multiple Linear Regression using Gradient Descent based approach

Python code: Multiple Linear Regression using sklearn

Python code: Ridge and Lasso Regression


![image](https://user-images.githubusercontent.com/67232573/113744291-74586b80-96b9-11eb-8fe4-c4dd5c9fb4e0.png)




# Logistic Regression

Logistic regression vs Linear Regression

Can we use Regression Mechanism for Classification?

Logistic Regression – Deriving the Formula

Logistic Regression for Multi-class Classification

Logistic Regression Decision Boundary

Python Code: Logistic regression on the titanic dataset- Part 1

Python Code: Logistic regression on the titanic dataset- Part 2

Python Code: Logistic regression on the titanic dataset- Part 3

Python Code: Logistic regression on the titanic dataset- Part 4

Python Code: Visualizing a logistic regression model


![image](https://user-images.githubusercontent.com/67232573/113744470-98b44800-96b9-11eb-9671-20962ef5b6d0.png)



![image](https://user-images.githubusercontent.com/67232573/113744512-a073ec80-96b9-11eb-9384-eec3b65e9fa3.png)




# Feature Selection

What is feature selection? Why feature selection?

Feature selection vs feature extraction

Feature subset selection using Filter based methods

More Filter based methods for feature selection

Wrapper Methods and their Comparison with Filter Methods

Wrapper Methods

Embedded Methods

Model based machine learning with regularization

Regularization using L2

Regularization using L1

Python Code: Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)

Python Code: Recursive Feature Elimination — wrapper

Python Code: Choosing important features (feature importance)

Python Code: Feature Selection using Variance Threshold


![image](https://user-images.githubusercontent.com/67232573/113744687-c6998c80-96b9-11eb-8a8d-13940bbbb5f0.png)


![image](https://user-images.githubusercontent.com/67232573/113744728-d1ecb800-96b9-11eb-9dde-7f04ef2354c0.png)




# Sequence Labeling

Introduction to Sequence Learning

Sequence Labeling as Classification

Probabilistic Sequence Models

Hidden Markov Model

Details about HMMs

Dishonest Casino Example of an HMM

Three Problems of an HMM

Decoding Problem of an HMM and the Viterbi Algorithm

Evaluation Problem of an HMM

The Forward Algorithm

The Backward Algorithm and the Posterior Decoding

The Learning Problem of an HMM, The Baum Welch Algorithm

Conditional Random Fields (CRFs)

Why prefer CRFs over HMMs?

Python code: Creating a simple Gaussian HMM

Python code: Learning a Gaussian HMM

Python code: Sampling from HMM

Python Code: Use CoNLL 2002 data to build a NER system: Understand the dataset

Python Code: Use CoNLL 2002 data to build a NER system: Define features

Python Code: Use CoNLL 2002 data to build a NER system: Learn and evaluate the CRF

Python Code: Use CoNLL 2002 data to build a NER system: Hyper-parameter Optimization

Python Code: Use CoNLL 2002 data to build a NER system: Feature Importances


# Clustering

Applications of Clustering

Understanding Distance

Basics of Clustering

Hierarchical (Agglomerative) clustering Part 1

Hierarchical (Agglomerative) clustering Part 2

K-means Algorithm example

K-means Algorithm details

Problems with K-means

Evaluation of cluster quality

Engineering issues with clustering

Soft clustering and EM algorithm example

Clustering summary

Python code: Kmeans Example

Python code: Kmeans on digits Example

Python code: Clustering for color compression

Mini Batch KMeans

Python code: Agglomerative Hierarchical Clustering

Ensemble Methods for Clustering: Problem Definition

Ensemble Methods for Clustering: Image Segmentation

Ensemble Methods for Clustering: Broad Approach

Ensemble Methods for Clustering: Finding Corresponding Clusters

Ensemble Methods for Clustering: Combining Corresponding Clusters


![image](https://user-images.githubusercontent.com/67232573/113744951-0496b080-96ba-11eb-98e7-4fad6bf9b4e4.png)



# Dimensionality Reduction using PCA and LDA

Why PCA?

PCA: A Layman’s Introduction

Understanding Matrix Transformations and Definition of Eigen Vectors

How is PCA Computed?

PCA Examples

Relationship between PCA, Curve Fitting and Entropy

Eigenfaces in OpenCV

Kernel PCA

Python Code: Compute PCA and show components

Python Code: PCA as dimensionality reduction

Python Code: PCA for visualization: Hand-written digits

Python Code: Eigenfaces

LDA

PCA vs LDA

2 class LDA

2 class LDA: Computing within and Between Class Scatter

2 class LDA Full Example

LDA for C classes

Limitations of LDA

Python Code: LDA on Wine dataset

Python Code: LDA from Scikit Learn on Iris dataset

Python Code: LDA on Iris dataset from scratch



![image](https://user-images.githubusercontent.com/67232573/113745240-242dd900-96ba-11eb-8095-ef352ab5a4e7.png)




# Architecting ML solutions

Machine Learning Process

Qualities of a Classifier

Technical Practical Issues in ML

Non-Technical Practical Issues in ML


![image](https://user-images.githubusercontent.com/67232573/113745609-47588880-96ba-11eb-8a38-46df5a215e83.png)



# ML case studies

Machine Learning for Healthcare – Part 1

Machine Learning for Healthcare – Part 2

Machine Learning for Internet Service Providers

Machine Learning for People Analytics

Machine Learning for Retail and Telecom – Part 1

Machine Learning for Retail and Telecom – Part 2

Machine Learning for Supply Chain Management

Machine Learning for Agriculture

Machine Learning for Education

Machine Learning for Transportation and self-driving cars

Machine Learning for Connected Cars

Machine Learning for Legal Domain – Part 1

Machine Learning for Legal Domain – Part 2

Machine Learning for Oil Industry

Machine Learning for Banking Domain – Part 1

Machine Learning for Banking Domain – Part 2

Machine Learning for Insurance

Machine Learning for Project Management

Machine Learning for Fashion Industry

Other use-cases of Machine Learning


![image](https://user-images.githubusercontent.com/67232573/113745874-8dade780-96ba-11eb-83a7-d6f358257ffc.png)


