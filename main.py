
'''
Course: TM10007 - Machine learning
Editors: Lishia Vergeer, Amy Roos, Maaike Pruijt, Hilde Roording.

Description: The aim of this code is to predict the tumor grade of glioma’s(high or low) before surgery, 
based on features extracted from a combination of four MRI images: 
T2-weighted, T2-weighted FLAIR and T1-weighted before and after injection of contrast agent.
'''

# General packages
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets as ds
from sklearn import decomposition
import seaborn

# Import code
from brats.load_data import load_data
from sklearn.model_selection import train_test_split

# scaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Classifiers
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn import metrics
from sklearn import feature_selection 
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import svm


def split_data(data_brats):

    """
    This function creates a panda dataframe and splits the data into test and train components.
    This is done with test_size variable and the function train_test_split from the sklearn module.
    Returns a train set with the data of 55% and a test set of 45% of the subjects.
    """

    data_features = pd.DataFrame(data=data_brats)
    data_train, data_test = train_test_split(data_features, test_size=0.45) # Nog bepalen wat test_size wordt
    print(f'data_train: {data_train}')
    print(f'data_test: {data_test}')
    return data_train, data_test

def no_none(data):
    '''
    Drop rows and colums with to many None. Threshold for minimum amount of non-None values in a row and a column is set on ... and ... respectively. 
    '''
    data_no_none = data.dropna() #axis=1, thresh=91) #, thresh=None, subset=None, inplace=False)
    print(data_no_none.head())
    print(data_no_none.size)
    return data_no_none

def split_xy(data_no_none):
    y = data_no_none_train.pop('label')
    X = data_no_none_train
    return y, X

def missing_data(data):
    '''
    Filling missing data
    '''

def feature_scale(data_train):
    '''
    Scale features
    '''
    # standard scaler
    scaler = StandardScaler()
    scaler.fit(data_train)
    X_scaled = scaler.transform(data_train)

    print(X_scaled)

    scaler_two = MinMaxScaler()
    scaler_two.fit(data_train)
    X_scaled_two = scaler_two.transform(data_train)

    print(X_scaled_two)

    scaler_three = RobustScaler()
    scaler_three.fit(data_train)
    X_scaled_three = scaler_three.transform(data_train)

    print(X_scaled_three)
    return X_scaled_three


def feature_transform(X_train, X_test, y_train, y_test):
    '''
    Transformation of features (PCA)
    '''
    # Waardes aanpassenn naar gescalde variant.    
    
    # Perform a PCA
    pca = decomposition.PCA(n_components=2)
    pca.fit(X_train) 
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Fit kNN
    knn = neighbors.KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train_pca, y_train)
    score_train = knn.score(X_train_pca, y_train)
    score_test = knn.score(X_test_pca, y_test)

    # Print result
    print(f"Training result: {score_train}")
    print(f"Test result: {score_test}")

# Uit voorbeeld, Not niet gereed:
def colorplot(clf, ax, x, y, h=100):
    '''
    Overlay the decision areas as colors in an axes.
    
    Input:
        clf: trained classifier
        ax: axis to overlay color mesh on
        x: feature on x-axis
        y: feature on y-axis
        h(optional): steps in the mesh
    '''
    # Create a meshgrid the size of the axis
    xstep = (x.max() - x.min() ) / 20.0
    ystep = (y.max() - y.min() ) / 20.0
    x_min, x_max = x.min() - xstep, x.max() + xstep
    y_min, y_max = y.min() - ystep, y.max() + ystep
    h = max((x_max - x_min, y_max - y_min))/h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    if len(Z.shape) > 1:
        Z = Z[:, 1]
    
    # Put the result into a color plot
    cm = plt.cm.RdBu_r
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    del xx, yy, x_min, x_max, y_min, y_max, Z, cm


if _name_ == "_main_":
    data_brats = load_data() 
    data_train, data_test = split_data(data_brats)

    #TRAIN
    data_no_none_train = no_none(data_train)
    y_train, X_train = split_xy(data_no_none_train)
    
    #TEST
    data_no_none_test = no_none(data_test)
    y_test, X_test = split_xy(data_no_none_test)
    
    
    feature_transform(y_train, X_train, y_test, X_test)
    # colorplot(clf, ax, x, y, h=100)