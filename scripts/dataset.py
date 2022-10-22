import numpy as np

from scripts.helpers import load_csv_data


class Dataset:
    def __init__(self, data_pth, data_type):
        self.data_pth = f"{data_pth}/{data_type}.csv"
        self.data_type = data_type

        self.data = []
        self.labels = []
        self.ids = []

        self.col_names = self.read_col_names()
        self.num_cols = len(self.col_names)

    def load_data(self):
        """Load the data from the csv file."""

        y, tX, ids = load_csv_data(self.data_pth)
        self.ids = ids
        self.labels = y
        self.data = tX

        self.data_imputation()
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
            "0": [0.0, 0.0, 0.0, 1.0],
            "1": [0.0, 0.0, 1.0, 0.0],
            "2": [0.0, 1.0, 0.0, 0.0],
            "3": [1.0, 0.0, 0.0, 0.0]
        }

        jet_num_col = self.col_names.index("PRI_jet_num")

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
