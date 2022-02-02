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
class Breast_Cancer_Wisconsin:
    """

    Breast Cancer Wisconsin Dataset
    ============
    Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
    They describe characteristics of the cell nuclei present in the image.
    1) Diagnosis (M = malignant, B = benign)

    Ten real-valued features are computed for each cell nucleus:

    a) radius (mean of distances from center to points on the perimeter)
    b) texture (standard deviation of gray-scale values)
    c) perimeter
    d) area
    e) smoothness (local variation in radius lengths)
    f) compactness (perimeter^2 / area - 1.0)
    g) concavity (severity of concave portions of the contour)
    h) concave points (number of concave portions of the contour)
    i) symmetry
    j) fractal dimension ("coastline approximation" - 1)

    The mean, standard error and "worst" or largest (mean of the three
    largest values) of these features were computed for each image,
    resulting in 30 features. For instance, field 3 is Mean Radius, field
    13 is Radius SE, field 23 is Worst Radius.

    All feature values are recoded with four significant digits.

    Missing attribute values: none

    Class distribution: 357 benign, 212 malignant
    """

    def __init__(self):
        # reading the dataset from source
        self.data = pd.read_csv("tab_automl/datasets/Breast Cancer Wisconsin.csv")
        # storing the columns overview
        self.columns = pd.Series([str(self.data[feature].dtype) for feature in self.data.columns])
        self.labels = None
        print("Populated the dataframe with data records...")

    def prepare_x_and_y(self,
                        feature_set_columns=["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"],
                        target_column="diagnosis"):
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
