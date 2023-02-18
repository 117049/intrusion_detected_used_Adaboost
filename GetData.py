import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler

class GetData:
    """
    pass in the original dataset path and return the processed pandas dataset
    default: label column is the last column, [:, -1]
    """

    def __init__(self, same_path="", sub_path={}):    
        self.data_init_utils=DataProcessingUtils()
        self.same_path=same_path
        self.sub_path=sub_path
        

    """NSL-KDD data"""
    def get_NSL_KDD_data_train(self, path=""):
        if len(path)==0:
            path=self.sub_path["NSL_KDD_train"]
        assert len(path)>0, 'data source path cannot be empty'
        data = pd.read_csv(self.same_path+path)
        return data

    def get_NSL_KDD_data_test(self, path=""):
        if len(path)==0:
            path=self.sub_path["NSL_KDD_test"]
        assert len(path)>0, 'data source path cannot be empty'
        data = pd.read_csv(self.same_path+path)
        return data
    
    
    """UNSW-NB15 data"""
    def get_UNSW_NB15_data_train(self, path=""):
        if len(path)==0:
            path=self.sub_path["UNSW_NB15_train"]
        assert len(path)>0, 'data source path cannot be empty'
        data = pd.read_csv(self.same_path+path)
        new_data = data.iloc[:,0:-1].copy()
        new_data = self.data_init_utils.encode_categorial(new_data)
        new_data = self.data_init_utils.encode_category_feature(new_data)
        return new_data

    def get_UNSW_NB15_data_test(self, path=""):
        if len(path)==0:
            path=self.sub_path["UNSW_NB15_test"]
        assert len(path)>0, 'data source path cannot be empty'
        data = pd.read_csv(self.same_path+path)
        new_data = data.iloc[:,0:-1].copy()
        new_data = self.data_init_utils.encode_categorial(new_data)
        new_data = self.data_init_utils.encode_category_feature(new_data)
        return new_data

    """common function"""
    def get_feature_num(self, data):
        return data.shape[1]-1

    def get_sample_num(self, data):
        return data.shape[0]

    def get_label_num(self, data):
        return len(set(data.iloc[:, -1]))

    def get_label(self, data):
        return np.unique(data.iloc[:, -1])

    def get_balance_factor(self, data):
        data = pd.DataFrame(data)
        label_set = np.unique(data.iloc[:, -1])
        balance_dict = {}
        for i in label_set:
            balance_dict[i] = len(data[data.iloc[:, -1] == i])
        return balance_dict

    def get_feature_info(self, data):
        data = pd.DataFrame(data)
        data_feature = data.iloc[:, 0:-1]
        feature_num = list(data_feature.select_dtypes(exclude=['object']).columns) #数值特征
        feature_category = list(filter(lambda x : x not in feature_num ,list(data_feature.columns))) #类别特征
        feature_info = {}
        feature_info["数值特征数量"] = len(feature_num)
        feature_info["类别特征数量"] = len(feature_category)
        feature_info["特征最大值"] = data_feature.max()
        feature_info["特征最小值"] = data_feature.min()
        feature_info["特征中位数"] = data_feature.median()
        feature_info["特征均值"] = data_feature.mean()
        return feature_info

    def get_feature_encode_data(self, data, mode='label_encoder'):
        """
        encode category features
        default: label column is the last column, [:, -1]
        """
        new_data = self.data_init_utils.encode_category_feature(data, mode)
        return new_data

    def get_standard_data(self, data):
        """
        normalize feature by standard scaler
        default: label column is the last column, [:, -1]
        """
        new_data = self.data_init_utils.standard_scaler_feature(data)
        return new_data


class DataProcessingUtils:
    """Initial data processing"""

    def __init__(self):
        pass
    
    def encode_categorial(self, data):
        new_data = data.copy()
        label_num = Counter(new_data.iloc[:, -1])
        print("data categotial info which need encode:" + str(label_num))
        new_data.iloc[:, -1] = LabelEncoder().fit_transform(np.array(data.iloc[:, -1]))
        new_label_info = Counter(new_data.iloc[:, -1])
        print("after encode, label info modified:" + str(new_label_info))
        return new_data
    
    def encode_category_feature(self, data, mode='label_encoder'):
        new_data = data.iloc[:, 0:-1].copy()
        feature_num = list(new_data.select_dtypes(exclude=['object']).columns) #数值特征
        feature_category = list(filter(lambda x : x not in feature_num, list(new_data.columns))) #类别特征
        
        for i in feature_category:
            unique_category = len(new_data[i].unique())
            print(str(i)+'的类别数量为：'+str(unique_category))

            if mode == 'label_encoder':
                new_data[i] = LabelEncoder().fit_transform(new_data[i])
            elif mode == 'one_hot_encoder':
                new_data[i] = OneHotEncoder().fit_transform(new_data[i])
            else:
                raise "mode is illegal"
        new_data = pd.concat([new_data, data.iloc[:, -1]], axis=1)
        return new_data

    def standard_scaler_feature(self, data):
        """
        normalization features by standard scaler
        default: label column is the last column, [:, -1]
        """
        df_x = data.iloc[:,0:-1]
        df_y = data.iloc[:,-1]
        x_columns = df_x.columns
        df_x = np.array(df_x)
        df_x = np.nan_to_num(df_x)
        scaler1 = StandardScaler().fit(df_x)
        df_x = scaler1.transform(df_x)
        df_x = pd.DataFrame(df_x, columns = x_columns)
        
        df_x = df_x.reset_index(drop=True)
        df_y = df_y.reset_index(drop=True)
        new_data = pd.concat([df_x,df_y],axis=1)
        return new_data