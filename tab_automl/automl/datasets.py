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
        if path.endswith(".txt"):
            self.data = pd.read_table(path, delimiter='\s')
        elif path.endswith(".json"):
            self.data = pd.read_json(path)
        elif path.endswith(".xlsx"):
            self.data = pd.read_excel(path)
        elif path.endswith(".sqlite"):
            table_name = input("table name :")
            db = sqlite3.connect(path)
            self.data = pd.read_sql_query(f'Select * from {table_name}', db)
        elif path.endswith(".csv"):
            self.data = pd.read_csv(path)
        else:
            print("File type not supported")
        # storing the columns overview
        self.columns = pd.Series([str(self.data[feature].dtype) for feature in self.data.columns])
        self.labels = None
        print(f"Populated the dataframe with data records...")

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
        print(f"X feature set and target feature has been split...")

        return x, y


class RegressionDataset:
    """

    Dataset Class for Regression.
    Currently supports single feature target only.

    """

    def __init__(self, path):
        # reading the dataset from source
        if path.endswith(".txt"):
            self.data = pd.read_table(path, delimiter='\s')
        elif path.endswith(".json"):
            self.data = pd.read_json(path)
        elif path.endswith(".xlsx"):
            self.data = pd.read_excel(path)
        elif path.endswith(".sqlite"):
            table_name = input("table name :")
            db = sqlite3.connect(path)
            self.data = pd.read_sql_query(f'Select * from {table_name}', db)
        elif path.endswith(".csv"):
            self.data = pd.read_csv(path)
        else:
            print("File type not supported")

        # storing the columns overview
        self.columns = pd.Series([str(self.data[feature].dtype) for feature in self.data.columns])
        print(f"Populated the dataframe with data records...")

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
        print(f"X feature set and target feature has been split...")

        return x, y
    
    
class ClusteringDataset:
    """

    Dataset Class for Clustering.
    Currently supports single feature target only.

    """

    def __init__(self, path):
        # reading the dataset from source
        if path.endswith(".txt"):
            self.data = pd.read_table(path, delimiter='\s')
        elif path.endswith(".json"):
            self.data = pd.read_json(path)
        elif path.endswith(".xlsx"):
            self.data = pd.read_excel(path)
        elif path.endswith(".sqlite"):
            table_name = input("table name :")
            db = sqlite3.connect(path)
            self.data = pd.read_sql_query(f'Select * from {table_name}', db)
        elif path.endswith(".csv"):
            self.data = pd.read_csv(path)
        else:
            print("File type not supported")
        # storing the columns overview
        self.columns = pd.Series([str(self.data[feature].dtype) for feature in self.data.columns])
        self.labels = None
        print(f"Populated the dataframe with data records...")


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
        print(f"Populated the dataframe with data records...")

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
        print(f"X feature set and target feature has been split...")

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
        print(f"Populated the dataframe with data records...")

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
        print(f"X feature set and target feature has been split...")

        return x, y
    
    
class Credit_Card_Customer_Data:
    """

    Credit_Card_Customer_Data Dataset
   ============

    This class will be an example for
    other clustering datasets or for
    custom datasets prepared on comma
    separated value format to be precise.

    """

    def __init__(self):
        # reading the dataset from source
        self.data = pd.read_csv("tab_automl/datasets/Credit_Card_Customer_Data.csv")
        # storing the columns overview
        self.columns = pd.Series([str(self.data[feature].dtype) for feature in self.data.columns])
        self.labels = None
        print(f"Populated the dataframe with data records...")

        """
        Defining x
        """

        x = self.data
        print(f"X feature set has been set...")

        return x

