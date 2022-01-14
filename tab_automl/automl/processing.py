"""
This file holds all codes for processing of raw data.
"""
import numpy as np
import os
import pandas as pd


class NullProcessing:

    """

    AutoML Class to process null
    values and features intelligently

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

    def integer_feature_processing(self, feature, drop_threshold=0.8, continuous_threshold=50):
        """
        Processes features with null values of integer data-types

        Args:
          feature: (string) name of the feature that has to be processed
          drop_threshold: (float) Minimum ratio of null and regular items to drop the feature
          continuous_threshold: (int) Minimum count of unique feature to be declared as continuous feature

        Returns:
            None

        """

        null_ratio = self.x[feature].isna().sum() / self.length
        if null_ratio >= drop_threshold:
            self.x.drop(feature, 1, inplace=True)
        else:
            unique_count = self.x[feature].nunique()
            # Checking if the feature is continuous or not
            if unique_count >= continuous_threshold:
                # calculating the median of the series
                median_value = np.nanmedian(self.x[feature])
                # filling the missing values with the median of the continuous series
                self.x[feature].fillna(median_value, inplace=True)
            elif unique_count >= continuous_threshold // 3:
                # calculating the mean of the series
                mean_value = np.nanmean(self.x[feature])
                # filling the missing values with the mean value of the series
                self.x[feature].fillna(mean_value, inplace=True)
            else:
                # calculating the mode
                mode_value = self.x[feature].dropna().value_counts().index[0]
                # filling the missing values with the mode of the integer series
                self.x[feature].fillna(mode_value, inplace=True)

    def run(self):
        # Iterating through features
        for feature in self.x.columns:
            # Checking if the column has any null value
            if self.x[feature].isnull().sum() > 0:
                # Checking whether the feature is integer type
                if self.x[feature].dtype == "int64":
                    self.integer_feature_processing(feature=feature)
                else:
                    # Will be updated soon
                    pass

        return self.x, self.y


class PreProcessing:
    """

    Overall PreProcessing class to preprocess raw
    data into a better understandable format.

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
            processed_dataframe.to_csv("processed_dataframe.csv", index=False)
            # Validating output path
            output_path = os.path.join(os.getcwd(), "processed_dataframe.csv")
            assert os.path.isfile(output_path), "Processed data saved on different location!"
            print(f"Processed data saved at {output_path}")
        else:
            # Will be updated soon
            pass

    def run(self):
        """

        Returns:
            x: (pandas.DataFrame) Processed feature set / Affecting features
            y: (pandas.DataFrame) Processed target feature

        """
        print("Initiating Preprocessing...")
        # Using Null Processor
        print("Going through null values and features...")
        null_dropper = NullProcessing(self.x, self.y)
        self.x, self.y = null_dropper.run()
        print("Null values and features processed...")

        print("Finishing Preprocessing...")

        return self.x, self.y
