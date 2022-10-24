from unicodedata import category
import numpy as np
from scripts.helpers import load_csv_data


class Dataset:
    def __init__(self, data_pth, data_type, imputation_method='mean', poly_degree=4):
        self.data_pth = f"{data_pth}/{data_type}.csv"
        self.data_type = data_type

        self.data = []
        self.labels = []
        self.ids = []
        self.category_data = []
        self.category_col_names = []

        self.poly_degree = poly_degree
        self.poly_data = []
        self.poly_col_names = None

        # after concatenating categorical variables
        self.poly_full_data = []
        self.poly_full_col_names = None

        self.col_names = self.read_col_names()
        self.num_cols = len(self.col_names)

        self.imputation_method = imputation_method

    def load_data(self):
        """Load the data from the csv file."""

        y, tX, ids = load_csv_data(self.data_pth)
        self.ids = ids
        self.labels = y
        self.data = tX

        self.category_feature()
        self.data_imputation()
        self.data_polynomial()
        self.data_normalization()
        self.filter_outliers()

    def read_col_names(self):
        """Read the column names from the csv file."""

        with open(self.data_pth, "r") as f:
            col_names = f.readline().strip().split(",")
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

        mean_data = np.mean(self.data, axis=0)
        self.data = self.data - mean_data
        self.data = self.data / np.std(self.data, axis=0)

    def category_feature(self):
        """Create new features based on the category feature."""
        jet_num_one_hot = {
            0: [0.0, 0.0, 0.0, 1.0],
            1: [0.0, 0.0, 1.0, 0.0],
            2: [0.0, 1.0, 0.0, 0.0],
            3: [1.0, 0.0, 0.0, 0.0]
        }

        jet_num_col = self.col_names.index("PRI_jet_num")
        self.category_col_names = ['jetnum3', 'jetnum2', 'jetnum1', 'jetnum0']

        # category val is in col 22
        category_data = self.data[:, 22]
        self.category_data = np.array(list(map(lambda x: jet_num_one_hot[x], category_data)))

        self.col_names = self.col_names + self.category_col_names
        self.data = np.c_[self.data, self.category_data]

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
        poly_data = []
        poly_col_names = []

        for i in range(self.data.shape[1]):
            col_name = self.col_names[i]
            col = self.data[:, i]

            poly = [col**j for j in range(1, degree+1)]
            poly_name = [col_name.replace('\n','')+'_'+str(j) for j in range(1, degree+1)]

            poly_data += poly
            poly_col_names += poly_name
# polynomial + original data
        self.poly_data = np.stack(poly_data)
        self.poly_data = self.poly_data.T
        self.poly_col_names = poly_col_names

# stack polynomial data + original data + categorical data
        self.poly_full_data = np.c_[self.poly_data, self.category_data]
        self.poly_full_col_names = self.poly_col_names + self.category_col_names
