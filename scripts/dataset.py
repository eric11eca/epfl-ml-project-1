import numpy as np

from proj1_helpers import load_csv_data


class Dataset:

    def __init__(self, data_pth, data_type):
        self.data_pth = f"{self.data_pth}/{self.data_type}.csv"
        self.data_type = data_type

        self.data = []

    def load_data(self):
        y, tX, ids = load_csv_data(self.data_pth)
        self.ids = ids
        self.labels = y
        self.data = tX

    def read_col_names(self):
        with open(self.data_pth, "r") as f:
            col_names = f.readline().strip().split(",")
        return col_names
