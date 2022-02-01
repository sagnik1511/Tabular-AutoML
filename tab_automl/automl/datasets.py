"""
This file holds all codes for custom and defined datasets.
"""
import sqlite3

import pandas as pd


class ClassificationDataset:
    """

    Dataset Class for Classification.
    Currently supports single feature target only.

    """

    def __init__(self, path):
        # reading the dataset from source
        if path[-4:] == ".txt":
            self.data = pd.read_table(path, delimiter='\s')
        elif path[-5:] == ".json":
            self.data = pd.read_json(path)
        elif path[-5:] == ".xlsx":
            self.data = pd.read_excel(path)
        elif path[-7:] == ".sqlite":
            table_name = input("table name :")
            db = sqlite3.connect(path)
            self.data = pd.read_sql_query(f'Select * from {table_name}', db)
        else:
            self.data = pd.read_csv(path)
        # storing the columns overview
        self.columns = pd.Series([str(self.data[feature].dtype) for feature in self.data.columns])
        self.labels = None
        print("Populated the dataframe with data records...")

    def prepare_x_and_y(self,
                        feature_set_columns,
                        target_column):
        """
        Function to segregate X and y from dataset

        Args:
          feature_set_columns: (List) Columns that'll be present on X.
          target_column: (Any) Column that'll be present on Y.

        Returns:
          x: (pandas.DataFrame) Feature set / Affecting features.
          y: (pandas) Target set / dependent feature.
        """

        x = self.data[feature_set_columns]
        y = self.data[[target_column]]
        self.labels = pd.Series(y.all()).unique()
        print("X feature set and target feature has been split...")

        return x, y


class RegressionDataset:
    """

    Dataset Class for Regression.
    Currently supports single feature target only.

    """

    def __init__(self, path):
        # reading the dataset from source
        if path[-4:] == ".txt":
            self.data = pd.read_table(path,delimiter='\s')
        elif path[-5:] == ".json":
            self.data=pd.read_json(path)
        elif path[-5:] == ".xlsx":
            self.data=pd.read_excel(path)
        elif path[-7:] == ".sqlite":
            table_name = input("table name :")
            db = sqlite3.connect(path)
            self.data = pd.read_sql_query(f'Select * from {table_name}', db)
        else:
            self.data = pd.read_csv(path)

        # storing the columns overview
        self.columns = pd.Series([str(self.data[feature].dtype) for feature in self.data.columns])
        print("Populated the dataframe with data records...")

    def prepare_x_and_y(self,
                        feature_set_columns,
                        target_column):
        """
        Function to segregate X and y from dataset

        Args:
          feature_set_columns: (List) Columns that'll be present on X.
          target_column: (Any) Column that'll be present on Y.

        Returns:
          2 pandas.DataFrame object i.e. X and Y
        """

        x = self.data[feature_set_columns]
        y = self.data[[target_column]]
        print("X feature set and target feature has been split...")

        return x, y


class Iris:
    """

    Iris Dataset
   ============

    This class will be an example for
    other classification datasets or for
    custom datasets prepared on comma
    separated value format to be precise.

    """

    def __init__(self):
        # reading the dataset from source
        self.data = pd.read_csv("tab_automl/datasets/Iris.csv")
        # storing the columns overview
        self.columns = pd.Series([str(self.data[feature].dtype) for feature in self.data.columns])
        self.labels = None
        print("Populated the dataframe with data records...")

    def prepare_x_and_y(self,
                        feature_set_columns=['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
                        target_column="Species"):
        """
        Function to segregate X and y from dataset.

        Args:
          feature_set_columns: (List) Columns that'll be present on X.
          target_column: (Any) Column that'll be present on Y.

        Returns:
          2 pandas.DataFrame object i.e. X and Y
        """

        x = self.data[feature_set_columns]
        y = self.data[[target_column]]
        self.labels = pd.Series(y.all()).unique()
        print("X feature set and target feature has been split...")

        return x, y


class Wine:
    """

    Wine Dataset
   ============

    This class will be an example for
    other regression datasets or for
    custom datasets prepared on comma
    separated value format to be precise.

    """

    def __init__(self):
        # reading the dataset from source
        self.data = pd.read_csv("tab_automl/datasets/wine.csv")
        # storing the columns overview
        self.columns_dtypes = pd.Series([str(self.data[feature].dtype) for feature in self.data.columns])
        print("Populated the dataframe with data records...")

    def prepare_x_and_y(self,
                        feature_set_columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'],
                        target_column='quality'):
        """
        Function to segregate X and y from dataset.

        Args:
          feature_set_columns: (List) Columns that'll be present on X.
          target_column: (Any) Column that'll be present on Y.

        Returns:
          2 pandas.DataFrame object i.e. X and Y
        """

        x = self.data[feature_set_columns]
        y = self.data[[target_column]]
        print("X feature set and target feature has been split...")

        return x, y
