from bayesianOpAda import BayesianOp
from GetData import GetData

path_dict={
        "NSL_KDD_train": "NSL_KDD_train.csv",
        "NSL_KDD_test": "NSL_KDD_test.csv",
        "UNSW_NB15_train": "UNSW_NB15_train.csv",
        "UNSW_NB15_test": "UNSW_NB15_test.csv",
    }

if __name__=="__main__":
    data_management = GetData(same_path="/root/intrusion_detection/Adaboost/data/", sub_path=path_dict)
    df_train = data_management.get_NSL_KDD_data_train()
    df_test = data_management.get_NSL_KDD_data_test()
    # df_train["class"] = df_train["class"].apply(lambda x : 0 if x == 0  else  1)
    # df_test["class"] = df_test["class"].apply(lambda x : 0 if x == 0  else  1)

    ba = BayesianOp(X=df_train.iloc[:, 0:-1], y=df_train.iloc[:, -1], test_X=df_test.iloc[:, 0:-1], test_y=df_test.iloc[:,-1])
    params_best, trials = ba.param_hyperopt(5)
    print(params_best)
    print(trials)

