'''
Course: TM10007 - Machine learning
Editors: Lishia Vergeer, Amy Roos, Maaike Pruijt, Hilde Roording.

Description: This is code which could also be used for splitting data. Not used in project. 
'''


def label_data(data_brats):
    '''
    Split data_brats in data and grade
    '''
    data_features = pd.DataFrame(data=data_brats)
    grade = data_features.pop('label')   

    data_train_enkel, data_test = train_test_split(data_brats, test_size=0.45) # Nog bepalen wat test_size wordt
    print(f'data_train_enkel: {data_train_enkel}')

    return data_features, grade


def split_data(data_features, grade):

    """
    This function splits the data into test and train components.
    This is done with test_size variable and the function train_test_split from the sklearn module.
    Returns a train set with the data of 55% and a test set of 45% of the subjects.
    """
    data_train, data_test, grade_train, grade_test = train_test_split(data_features, grade, test_size=0.45) # Nog bepalen wat test_size wordt
    print(f'data_train: {data_train}')
    return data_train, data_test, grade_train, grade_test