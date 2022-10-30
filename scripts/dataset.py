import numpy as np

from scripts.helpers import load_csv_data


class Dataset:
    def __init__(self, data_pth, data_type, imputation="median", poly_degree=4):
        self.data_pth = f"{data_pth}/{data_type}.csv"
        self.data_type = data_type
        self.poly_degree = poly_degree
        self.imputation = imputation

    def load_data(self, poly=False, outliers=False, discard_cols=None):
        """Load the data from the csv file."""
        if discard_cols is None:
            discard_cols = ['prediction', 'id', 'pri_jet_num']

        label_map = {1: 1, -1: 0}

        self.orig_col_name = self.read_col_names()
        self.feature_col_ids = [i for i, col in enumerate(
            self.orig_col_name, start=0) if col.lower() not in discard_cols]
        self.feature_col_names = [
            col for col in self.orig_col_name if col.lower() not in discard_cols]

        y, tX, ids = load_csv_data(self.data_pth)
        self.ids = ids
        self.tX = tX
        self.data = tX[:, self.feature_col_ids]
        self.labels = np.array(list(map(lambda y: label_map[y], y)))

        self.data_imputation(method=self.imputation)
        self.full_data, self.full_feature_names = self.build_category_feature()

        if poly:
            self.full_data = self.poly_data
            self.full_feature_names = self.poly_feature_names

        self.sanity_check()

        self.data_normalization()

        if self.data_type == 'train' and not outliers:
            self.filter_outliers()
        

    def sanity_check(self):
        """Check if the data is valid."""
        print('=== original columns ===')
        print(f'original data shape: {self.tX.shape}')
        print(self.orig_col_name)
        print()
        print('=== feature columns ===')
        print(f'feature data shape: {self.data.shape}')
        print(self.feature_col_names)
        print()
        print('=== categorical columns ===')
        print(f'categorical data shape: {self.full_data.shape}')
        print(self.full_feature_names)
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
                    self.data[100][self.data[100] == -
                                   999] = self.cat_mean[self.labels[100]][self.data[100] == -999]
            elif method == 'category_median':
                self.get_cat_mean()
                for row in range(self.data.shape[0]):
                    self.data[100][self.data[100] == -
                                   999] = self.cat_median[self.labels[100]][self.data[100] == -999]
            else:
                col_data[col_data == -999.0] = 0.0
                col_data[np.isnan(col_data)] = 0.0

    def data_normalization(self):
        """Normalize the data, zero-mean and standardization."""
        out_cols = [29, 30, 31, 32]
        in_cols = [i for i in range(np.shape(self.full_data)[1]) if i not in out_cols]
        extracted = self.full_data[:, in_cols]

        mean_data = np.mean(extracted, axis=0)
        extracted = extracted - mean_data
        extracted = extracted / np.std(extracted, axis=0)

        self.full_data = np.c_[extracted, self.category_data]

    def build_category_feature(self):
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
        self.category_data = np.array(
            list(map(lambda x: jet_num_one_hot[x], category_data)))

        full_col_names = self.feature_col_names + self.category_col_names
        full_data = np.c_[self.data, self.category_data]

        return full_data, full_col_names

    def filter_outliers(self, m=10):
        """
        Filter out outliers over mean +/- m * std>
        """
        for i in range(self.full_data.shape[1]):
            delta = abs(self.full_data[:, i] - np.mean(self.full_data[:, i]))
            mdev = m * np.std(self.full_data[:, i])
            self.full_data = self.full_data[delta < mdev]
            self.labels = self.labels[delta < mdev]
            self.ids = self.ids[delta < mdev]

            assert self.labels.shape[0] == self.full_data.shape[0]
            assert self.ids.shape[0] == self.full_data.shape[0]

    def get_cat_mean(self):
        """Get the mean of each category."""
        NAN_VALUE = -999
        # 'category_mean', 'category_median'
        self.cat_mean = {}
        self.cat_median = {}
        self.cat_ids = {cat: [i for i in range(
            len(self.labels)) if self.labels[i] == cat] for cat in np.unique(self.labels)}

        for col in range(self.data.shape[1]):
            for cat, ids in self.cat_ids.items():
                col_data = self.data[:, col][ids]
                if cat not in self.cat_mean.keys():
                    self.cat_mean[cat] = np.zeros(self.data.shape[1])
                    self.cat_median[cat] = np.zeros(self.data.shape[1])
                self.cat_mean[cat][col] = np.mean(
                    col_data[col_data != NAN_VALUE])
                self.cat_median[cat][col] = np.median(
                    col_data[col_data != NAN_VALUE])

    def data_polynomial(self, degree=None):
        """Create polynomial features."""
        degree = degree if degree else self.poly_degree

        if degree <= 1 :
            full_data = np.c_[self.data, self.category_data]
            full_col_names = self.feature_col_names + self.category_col_names

            return full_data, full_col_names

        poly_data = []
        poly_col_names = []

        for i in range(len(self.feature_col_ids)):
            col_name = self.feature_col_names[i]
            col = self.data[:, i]

            poly = [col**j for j in range(1, degree+1)]
            poly_name = [col_name.replace('\n', '')+'_'+str(j)
                         for j in range(1, degree+1)]

            poly_data += poly
            poly_col_names += poly_name

        # polynomial + original data
        poly_data = np.stack(poly_data)
        poly_data = poly_data.T

        # stack polynomial data + original data + categorical data
        poly_full_data = np.c_[poly_data, self.category_data]
        poly_full_col_names = poly_col_names + self.category_col_names

        return poly_full_data, poly_full_col_names
