import operator
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def get_train_test_sets(data, targets):
    encoder = LabelEncoder()
    encoder.fit(targets)
    y_enc = encoder.transform(targets)  # Numerical encoding of identities
    X_train, X_test, y_train, y_test = train_test_split(data, y_enc, test_size=0.30, random_state=42)
    return X_train, X_test, y_train, y_test


# Convert the training set into tuples so that it is easier to create the databases suitable for authentication
def create_tuples_list(X_train, y_train):
    tuples_list = []
    for i in range(len(X_train)):
        tuples_list.append((y_train[i], X_train[i]))
        tuples_list = sorted(tuples_list, key=operator.itemgetter(0))  # sort by value - feature distance

    return tuples_list


def prepare_dataset_authentication(X_train, y_train):
    '''

    :param tuples_list:
    :return:
    '''
    authentication_databases = {}
    tuples_list = create_tuples_list(X_train, y_train)
    class_name = tuples_list[0][0]  # inital class name
    last_key = tuples_list[-1][0]
    tmp_list = []

    for i in range(len(tuples_list)):
        if class_name != tuples_list[i][0]:
            authentication_databases[class_name] = tmp_list
            tmp_list = []
        tmp_list.append(tuples_list[i])
        class_name = tuples_list[i][0]

        if last_key == class_name:
            authentication_databases[class_name] = tmp_list

    return authentication_databases


def get_data_statistics(faces):
    le = LabelEncoder()
    targets = le.fit_transform(faces.target)
    n_classes = np.unique(targets).shape[0]
    n_samples, h, w = faces.images.shape
    n_features = faces.data.shape[1]

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)
