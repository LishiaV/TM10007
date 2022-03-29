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
    #print(f'data_train: {data_train}')
    #print(f'data_test: {data_test}')
    return data_train, data_test

def no_none(data):
    '''
    Deleting columns with NaN or filling them.
    '''
    # Inzicht in data
    print(f'OVERZICHT: {data.isnull().sum()}')

    # If the total number of NaN observations in a column are greater than 40%, delete the entire column.
    perc = 40.0
    min_count = int(((100-perc)/100)*data.shape[0] + 1)
    data_dropcolumn = data.dropna(axis=1, thresh=min_count)
    #print(data_dropcolumn)
    #print(data_dropcolumn.size)

    # fill the NaN observations.
    data_fill = data_dropcolumn.fillna(data_dropcolumn.median())
    #print(data_fill)
    #print(data_fill.size)

    # Inzicht in data
    print(f'OVERZICHT NONONE: {data_fill.isnull().sum()}')
    return data_fill

def split_xy(data_no_none):
    '''
    Split in X (data) and y (label)
    '''
    y = data_no_none.pop('label')
    X = data_no_none
    return y, X

def feature_scale(data_train):
    '''
    Scale features
    '''
    # standard scaler
    scaler = StandardScaler()
    scaler.fit(data_train)
    X_scaled = scaler.transform(data_train)
    print(X_scaled)
    
    # minmax scaler
    scaler_two = MinMaxScaler()
    scaler_two.fit(data_train)
    X_scaled_two = scaler_two.transform(data_train)
    print(X_scaled_two)
    
    # robustscaler
    scaler_three = RobustScaler()
    scaler_three.fit(data_train)
    X_scaled_three = scaler_three.transform(data_train)
    print(X_scaled_three)
    return X_scaled_two


def feature_transform(X_train, X_test):
    '''
    Transformation of features (PCA)
    '''
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
    

if __name__ == "__main__":
    data_brats = load_data() 
    data_train, data_test = split_data(data_brats)
    data_no_none_train = no_none(data_train)
    y_train, X_train = split_xy(data_no_none_train)
    #X_scale = feature_scale(X_train)
    #feature_transform(y_train, X_scale)


    


