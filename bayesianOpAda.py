"""贝叶斯参数优化"""

import numpy as np
from hyperopt import hp, fmin, tpe, Trials, partial
from hyperopt.early_stop import no_progress_loss
from Adaboost import Adaboost
from evaluate import ConfusionMatrix
import pandas as pd
import utils

class BayesianOp:
    def __init__(self, X, y=None, test_X=None, test_y=None):
        self.X=np.array(X)
        self.y=np.array(y)
        self.test_X=np.array(test_X)
        self.test_y=np.array(test_y)
        self.params=None
        self.num=0
        self.table=utils.table_evaluate()
        self.max_evals=0

    def hyperopt_objective(self, params):
    #定义评估器
    #需要搜索的参数需要从输入的字典中索引出来
    #不需要搜索的参数，可以是设置好的某个值
    #在需要整数的参数前调整参数类型

        Ada = Adaboost(n_estimators = int(params["n_estimators"])
                ,max_depth = int(params["max_depth"])
                ,max_features = int(params["max_features"])
                ,min_samples_split = int(params["min_samples_split"])
                ,min_weight_fraction_leaf = params["min_weight_fraction_leaf"]
                ,learning_rate = params["learning_rate"]
                ,random_state=2023)
        self.num += 1
        label = Ada.fit_transform(Y_train=self.y, X_train=self.X, X_test=self.test_X)

        evaluate = ConfusionMatrix()
        acc, bac, kappa, table = evaluate.evaluate(pred=label, labels=self.test_y)
        
        self.table.put_table(acc=acc, bac=bac, kappa=kappa)
        self.table.put_others_by_label(table)
        if self.num == self.max_evals:
            self.table.get_table().to_csv("/root/Adaboost_result/NSL_KDD_Baye_result.csv", header=True)
            for i in range(len(self.table.get_other_table_list())):
                self.table.get_other_table_list()[i].to_csv("/root/Adaboost_result/NSL_KDD_class_{}_Baye_result.csv".format(i), header=True)
        return -bac

    def param_hyperopt(self, max_evals=100, params=None):
        self.max_evals=max_evals
        if params==None:
            params = {'n_estimators': hp.quniform("n_estimators",20,300,1)
                     , 'max_depth': hp.quniform("max_depth",2,8,1)
                     , "max_features": hp.quniform("max_features",2,40,1)
                     , "min_samples_split": hp.quniform("min_samples_spli",2,10,1)
                     , "min_weight_fraction_leaf": hp.quniform("min_weight_fraction_leaf",0.0, 0.5, 0.1)
                     , "learning_rate": hp.quniform("learning_rate",0.001,1,0.001)
                    }
        self.params = params
        #保存迭代过程
        trials = Trials()
        
        #设置提前停止
        early_stop_fn = no_progress_loss(100)
        
        #定义代理模型
        algo = partial(tpe.suggest, n_startup_jobs=10, n_EI_candidates=30)
        params_best = fmin(self.hyperopt_objective #目标函数
                        , space = self.params #参数空间
                        , algo = algo #代理模型
                        , max_evals = max_evals #迭代次数
                        , verbose=True
                        , trials = trials
                        , early_stop_fn = early_stop_fn
                        )
        
        #打印最优参数，fmin会自动打印最佳分数
        print("\n","\n","best params: ", params_best,
            "\n")
        return params_best, trials
