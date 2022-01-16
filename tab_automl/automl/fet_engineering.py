"""
This files holds all codes for feature engineering on processed data.
"""
import numpy as np
import os
import pandas as pd


class Encode:
    """

    AutoML Class to process
    to encode features

    """

    def __init__(self, x, y):
        """
        Args:
          x: (pandas.DataFrame) Feature set / Affecting features
          y: (pandas.Dataframe) Target set / dependent feature

        """
        self.x = x
        self.y = y
        assert x.shape[0] == y.shape[0], "Data shape mismatched!"
        self.length = self.x.shape[0]

    def encode_single_feature(self, feature, hot_encode_threshold=5, least_present_unique_item_ratio=0.1):
        """
        Args:
            feature: (string) name of the feature that has to be processed
            hot_encode_threshold: (int) Maximum number of unique items to decide the encoding technique
            least_present_unique_item_ratio: (float) Minimum ratio of least present unique number in feature

        Returns:
            None

        """

        unique_items = self.x[feature].unique()
        least_present_unique_item_count = self.x[feature].value_counts().values[-1]
        if len(unique_items) <= hot_encode_threshold or \
                (least_present_unique_item_count / self.length) >= least_present_unique_item_ratio:
            # perform one hot encoding
            encoded_feature = pd.get_dummies(self.x[feature])
            # dropping the parent feature after encoding
            self.x.drop(feature, 1, inplace=True)
            # adding the encoded feature set to x
            self.x = pd.concat([self.x, encoded_feature], axis=1)
        else:
            # perform label encoding
            for index, item in enumerate(unique_items):
                # replacing the unique items with integer values
                self.x[feature].replace(item, index, inplace=True)

    def run(self):
        """
        Returns:
            x: (pandas.DataFrame) Feature set / Affecting features
            y: (pandas.Dataframe) Target set / dependent feature
        """

        # Iterating through features
        for feature in self.x.columns:
            if self.x[feature].dtype == "object":
                # Encoding the feature items if the feature is categorical
                self.encode_single_feature(feature=feature)

        return self.x, self.y


class FeatureEngineering:
    """

    Overall feature engineering AutoML
    Class for tabular datasets.

    Returns:
        Updated feature set(x) and target feature (y)

    """

    def __init__(self, x, y):
        """
        Args:
          x: (pandas.DataFrame) Feature set / Affecting features
          y: (pandas.Dataframe) Target set / dependent feature

        """
        self.x = x
        self.y = y
        assert x.shape[0] == y.shape[0], "Data shape mismatched!"

    def save_data(self, save_format="csv"):
        """
        Writes the processed data into
        a file on respective format

        Args:
            save_format: (Any) the required format on which the files will be saved.

        Returns:
            processed_dataframe: (pandas.DataFrame) Processed feature set / Affecting features
        """

        # Joining the x and y feature set to prepare full dataframe
        processed_dataframe = pd.concat([self.x, self.y], axis=1)
        if save_format == "csv":
            # Writing to csv
            processed_dataframe.to_csv("feature_engineered_dataframe.csv", index=False)
            # Validating output path
            output_path = os.path.join(os.getcwd(), "feature_engineered_dataframe.csv")
            assert os.path.isfile(output_path), "Processed data saved on different location!"
            print(f"Feature engineered data saved at {output_path}")
        else:
            # Will be updated soon
            pass

    def run(self):
        """

        Returns:
            x: (pandas.DataFrame) Processed feature set / Affecting features
            y: (pandas.DataFrame) Processed target feature

        """
        print("Initiating Feature Engineering...")
        # Using Encoder
        print("Encoding features...")
        encoder = Encode(self.x, self.y)
        self.x, self.y = encoder.run()
        print("Encoding finished...")

        print("Finishing Feature Engineering...")

        return self.x, self.y
