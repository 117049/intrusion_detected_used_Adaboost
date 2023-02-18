"""基于参数优化的Adaboost算法，用于网络入侵检测系统"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from multi_ada import AdaBoostClassifier as Ada

class Adaboost:
    """
    parameters: 
    n_estimators: integer, optional(default=50)
        The maximum number of estimators

    learning_rate: float, optional(default=1)

    algorithm: {'SAMME','SAMME.R'}, optional(default='SAMME.R')
        SAMME.R uses predicted probabilities to update wights, while SAMME uses class error rate

    random_state: int or None, optional(default=None)
    """
    def __init__(self, n_estimators=50, learning_rate=0.1, algorithm='SAMME.R', random_state=2023, min_samples_split=2,
                    min_weight_fraction_leaf=0.0, max_features=20, max_depth=5):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.algorithm = algorithm
        self.min_samples_split=min_samples_split
        self.min_weight_fraction_leaf=min_weight_fraction_leaf
        self.max_features=max_features
        self.max_depth=max_depth
        
    """ ADABOOST IMPLEMENTATION ================================================="""
    def fit_transform(self, Y_train, X_train, X_test):
        """
        param: U: the initialize weight of X_train
        """

        adaboost_result = Ada(
            base_estimator=DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split, criterion="gini",
                                                    splitter="best", min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                                    max_features=self.max_features),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
            random_state=self.random_state)
        adaboost_result.fit(X_train, Y_train)
        
        prob = adaboost_result.predict_proba(X_test)
        label = np.argmax(prob, axis=1)
        
        return label