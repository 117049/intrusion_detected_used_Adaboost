import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import numpy as np

def split_train_test(X, Y=None, size=0.2):
    """
    split dataset to test and train subsets
    """
    assert size >0 and size <1, 'the radio of test subset shoule in (0, 1)'
    if(Y == None):
        df = pd.DataFrame(X)
    else:
        X = pd.DataFrame(X)
        Y = pd.DataFrame(Y)
        df = pd.concat(X, Y, axis=1)
    # split into training and test set
    train, test = train_test_split(df, test_size=size)
    train_X, train_Y = train.iloc[:, :-1], train.iloc[:, -1]
    test_X, test_Y = test.iloc[:, :-1], test.iloc[:, -1]
    return train_X, train_Y, test_X, test_Y

def get_predict_acc_result(pred, Y, str=None):
    """show acc indication of result"""
    acc = sum(pred == Y) / float(len(Y))
    print('{} acc: {}'.format(str, acc))
    return acc

def get_predict_bac_result(pred, Y, str=None):
    """show balanced accuracy score"""
    bca = balanced_accuracy_score(Y, pred)
    print('{} bac: {}'.format(str, bca))
    return bca

def get_max_index(array, axis=0):
    res = np.argmax(array, axis=axis)
    return res

def get_mode(array):
    assert len(array) != 0, "in get mode function, array cannot be empty"
    return max(array, key=lambda v: array.count(v))

class table_evaluate:
    def __init__(self):
        self.tabel=pd.DataFrame(columns=["ACC", "BCA", "KAPPA"])
        self.Mluti_class=[]

    def put_table(self, acc, bac, kappa):
        self.tabel.loc[len(self.tabel)] = [acc, bac, kappa]

    def put_others_by_label(self, table):
        if len(self.Mluti_class)==0:
            for i in range(len(table)):
                self.Mluti_class.append(pd.DataFrame(columns=["Precision", "Recall", "Specificity"]))
        for i in range(len(table)):
            self.Mluti_class[i].loc[len(self.Mluti_class[i])] = table[i]

    def get_table(self):
        return self.tabel

    def get_other_table_list(self):
        return self.Mluti_class