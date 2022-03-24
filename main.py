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
import seaborn

# Import code
from brats.load_data import load_data
from sklearn.model_selection import train_test_split

# Classifiers
from sklearn import metrics
from sklearn.decomposition import PCA

# scaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def split_data(data_brats):

    """
    This function creates a panda dataframe and splits the data into test and train components.
    This is done with test_size variable and the function train_test_split from the sklearn module.
    Returns a train set with the data of 55% and a test set of 45% of the subjects.
    """

    data_features = pd.DataFrame(data=data_brats)
    data_train, data_test = train_test_split(data_features, test_size=0.45) # Nog bepalen wat test_size wordt
    return data_train, data_test


def feature_scale(data_train):
    '''
    Scale features
    '''

    return data_scaled


def convert_data(data_scaled):
    '''
    Drop rows and colums with to many None. Threshold for minimum amount of non-None values in a row and a column is set on ... and ... respectively. 
    '''
    data_no_none = data_scaled.dropna(axis=1) # Threshold to be determined with thresh = xx

    data_replace = data_no_none.replace(['LGG', 'GBM'], [0, 1])

    print(data_replace.head)


    return data_replace

def feature_scale(data_train):
    '''
    Scale features
    '''
    # standard scaler
    scaler = StandardScaler()
    scaler.fit(data_train)
    X_scaled = scaler.transform(data_train)

    print(X_scaled)
    return
'''
    
    scaler_two = MinMaxScaler()
    scaler_two.fit(data_train)
    X_scaled_two = scaler_two.transform(data_train)

    print(X_scaled_two)

    scaler_three = RobustScaler()
    scaler_three.fit(data_train)
    X_scaled_three = scaler_three.transform(data_train)

    print(X_scaled_three)
    '''


def feature_selection():
    '''
    Selection of features
    '''


def feature_transform():
    '''
    Transformation of features (PCA)
    '''


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


if __name__ == "__main__":
    data_brats = load_data()
    data_train, data_test = split_data(data_brats)
    data_converted = convert_data(data_train)
    # feature_scale(data_converted)

    # feature_selection()
    # feature_transform()
    # colorplot(clf, ax, x, y, h=100)


    


