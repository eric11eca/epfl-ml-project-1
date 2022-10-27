import numpy as np

from unicodedata import category
from scripts.helpers import load_csv_data


class Dataset:
    def __init__(self, data_pth, data_type, imputation_method='mean', poly_degree=4):
        self.data_pth = f"{data_pth}/{data_type}.csv"
        self.data_type = data_type
        self.poly_degree = poly_degree
        self.imputation_method = imputation_method

    def load_data(self, discard_cols=None):
        """Load the data from the csv file."""
        if discard_cols is None:
            discard_cols = ['prediction', 'id', 'pri_jet_num']

        self.labels, self.tX, self.ids = load_csv_data(self.data_pth)
        self.orig_col_name = self.read_col_names()
        self.jet_num_col = self.orig_col_name.index("PRI_jet_num")

        # Discard id, prediction and jet_num categorical cols
        self.feature_col_ids = [i for i, col in enumerate(self.orig_col_name, start=0) if col.lower() not in discard_cols]
        self.feature_col_names = [col for col in self.orig_col_name if col.lower() not in discard_cols]
        
        # preserved feature
        self.data = self.tX[:, self.feature_col_ids]
        
        # add categorical feature
        self.categorical_data, self.categorical_feature_names = self.category_feature()

        self.data_imputation()

        # add polynomial feature
        self.poly_data, self.poly_feature_names = self.data_polynomial()

        # Sanity Check
        self.sanity_check()

        self.data_normalization()
        self.filter_outliers()

    def sanity_check(self):
        print('=== original columns ===')
        print(f'original data shape: {self.tX.shape}')
        print(self.orig_col_name)
        print()
        print('=== feature columns ===')
        print(f'feature data shape: {self.data.shape}')
        print(self.feature_col_names)
        print()
        print('=== categorical columns ===')
        print(f'categorical data shape: {self.categorical_data.shape}')
        print(self.categorical_feature_names)
        print()
        print('=== poly columns ===')
        print(f'polynomial data shape: {self.poly_data.shape}')
        print(self.poly_feature_names)
        print()

    def read_col_names(self):
        """Read the column names from the csv file."""

        with open(self.data_pth, "r") as f:
            col_names = f.readline().strip().split(",")[2:]
        return col_names

    def data_imputation(self, method="median"):
        """Impute the missing values in the data."""

        for col in range(self.data.shape[1]):
            col_data = self.data[:, col]
            if method == "mean":
                col_data[col_data == -
                         999.0] = np.nanmean(col_data[col_data != -999.0])
                col_data[np.isnan(col_data)] = np.nanmean(col_data)
            elif method == "median":
                col_data[col_data == -
                         999.0] = np.nanmedian(col_data[col_data != -999.0])
                col_data[np.isnan(col_data)] = np.nanmedian(col_data)
            elif method == 'category_mean':
                self.get_cat_mean()
                for row in range(self.data.shape[0]):
                    self.data[100][self.data[100]==-999]= self.cat_mean[self.labels[100]][self.data[100]==-999]
            elif method == 'category_median':
                self.get_cat_mean()
                for row in range(self.data.shape[0]):
                    self.data[100][self.data[100]==-999]= self.cat_median[self.labels[100]][self.data[100]==-999]
            else:
                col_data[col_data == -999.0] = 0.0
                col_data[np.isnan(col_data)] = 0.0
                
    def data_normalization(self):
        """Normalize the data, zero-mean and standardization."""

        mean_data = np.mean(self.poly_data, axis=0)
        self.poly_data = self.poly_data - mean_data
        self.poly_data = self.poly_data / np.std(self.poly_data, axis=0)

    def category_feature(self):
        """Create new features based on the category feature."""
        jet_num_one_hot = {
            0: [0.0, 0.0, 0.0, 1.0],
            1: [0.0, 0.0, 1.0, 0.0],
            2: [0.0, 1.0, 0.0, 0.0],
            3: [1.0, 0.0, 0.0, 0.0]
        }

        self.category_col_names = ['jetnum3', 'jetnum2', 'jetnum1', 'jetnum0']
        # category val is in col 22
        category_data = self.tX[:, 22]
        self.category_data = np.array(list(map(lambda x: jet_num_one_hot[x], category_data)))

        categorical_feature_names = self.feature_col_names + self.category_col_names
        data = np.c_[self.data, self.category_data]
        return data, categorical_feature_names

    def filter_outliers(self, m=10):
        """
        Filter out outliers over mean +/- m * std>
        """
        for i in range(self.data.shape[1]):
            delta = abs(self.data[:, i] - np.mean(self.data[:, i]))
            mdev = m * np.std(self.data[:, i])
            self.data = self.data[delta < mdev]
            self.labels = self.labels[delta < mdev]
            self.ids = self.ids[delta < mdev]

            assert self.labels.shape[0] == self.data.shape[0]
            assert self.ids.shape[0] == self.data.shape[0]

    def get_cat_mean(self):
        NAN_VALUE = -999
        # 'category_mean', 'category_median'
        self.cat_mean = {}
        self.cat_median = {}
        self.cat_ids = {cat:[i for i in range(len(self.labels)) if self.labels[i]==cat] for cat in np.unique(self.labels)}


        for col in range(self.data.shape[1]):
            for cat, ids in self.cat_ids.items():
                col_data = self.data[:, col][ids]
                if cat not in self.cat_mean.keys():
                    self.cat_mean[cat] = np.zeros(self.data.shape[1])
                    self.cat_median[cat] = np.zeros(self.data.shape[1])
                self.cat_mean[cat][col]=np.mean(col_data[col_data!=NAN_VALUE])
                self.cat_median[cat][col]=np.median(col_data[col_data!=NAN_VALUE])

    def data_polynomial(self, degree=None):
        degree = degree if degree else self.poly_degree
        if degree <= 1:
            return self.categorical_data, self.categorical_feature_names
         
        poly_data = []
        poly_col_names = []

        for idx in range(len(self.feature_col_ids)):
            col_name = self.feature_col_names[idx]
            col = self.data[:, idx]

            poly = [col**j for j in range(1, degree+1)]
            poly_name = [col_name.replace('\n','')+'_'+str(j) for j in range(1, degree+1)]

            poly_data += poly
            poly_col_names += poly_name
        # polynomial + original data
        poly_data = np.stack(poly_data)
        poly_data = poly_data.T
        # poly_feature_names = poly_col_names

        # stack polynomial data + original data + categorical data
        poly_data = np.c_[poly_data, self.category_data]
        poly_feature_names = poly_col_names + self.category_col_names
        return poly_data, poly_feature_names
