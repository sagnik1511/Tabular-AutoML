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

    def numerical_feature_processing(self, feature, feature_drop_threshold=0.8, continuous_threshold=50):
        """
        Processes features with null values of integer / float / double data-types

        Args:
          feature: (string) name of the feature that has to be processed
          feature_drop_threshold: (float) Minimum ratio of null and regular items to drop the feature
          continuous_threshold: (int) Minimum count of unique feature to be declared as continuous feature

        Returns:
            None

        """

        null_ratio = self.x[feature].isna().sum() / self.length
        # Checking if null value ratio is higher than expected
        if null_ratio >= feature_drop_threshold:
            # Dropping the whole feature
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
                # Checking if the most abundant feature is null
                if mode_value == np.nan:
                    # Updating the mode value with most abundant regular item
                    mode_value = self.x[feature].dropna().value_counts().index[1]
                # filling the missing values with the mode of the integer series
                self.x[feature].fillna(mode_value, inplace=True)

    def categorical_feature_processing(self, feature, feature_drop_threshold=0.6, row_drop_threshold=0.3):
        """
        Processes features with null values of categorical type

        Args:
          feature: (string) name of the feature that has to be processed
          feature_drop_threshold: (float) Minimum ratio of null and regular items to drop the feature
          row_drop_threshold: (float) Minimum ratio of null and regular items to drop the rows with null feature

        Returns:
            None

        """

        null_ratio = self.x[feature].isna().sum() / self.length
        if null_ratio >= feature_drop_threshold:
            self.x.drop(feature, 1, inplace=True)
        elif null_ratio >= row_drop_threshold:
            # stored the feature names of x and y so that they can be retrieved again.
            x_features = self.x.columns
            y_feature = self.y.columns

            # Joined and prepare the whole dataframe
            joined_dataframe = pd.concat([self.x, self.y], axis=1)
            # Dropped all extra null values
            joined_dataframe.dropna(subset=[feature], inplace=True)
            # Retrieved X and y again
            self.x = joined_dataframe[x_features]
            self.y = joined_dataframe[y_feature]
        else:
            # calculating the mode
            abundant_item = self.x[feature].value_counts().index[0]
            # Checking if the most abundant feature is null
            if abundant_item == np.nan:
                # Updating the mode value with most abundant regular item
                abundant_item = self.x[feature].value_counts().index[1]
            # filling the missing values with the mode of the integer series
            self.x[feature].fillna(abundant_item, inplace=True)

    def run(self):
        # Iterating through features
        for feature in self.x.columns:
            # Checking if the column has any null value
            if self.x[feature].isnull().sum() > 0:
                print(f"{feature} has null values, total count : {self.x[feature].isnull().sum()} .")
                # Checking whether the feature is categorical
                if self.x[feature].dtype == "object":
                    self.categorical_feature_processing(feature=feature)
                else:
                    # Processing on the other data type features
                    self.numerical_feature_processing(feature=feature)
                print(f"Null values at {feature} processed...")

        # Checking if there's any null value on target feature
        if self.y.all().isnull().sum() > 0:
            # stored the feature names of x and y so that they can be retrieved again.
            x_features = self.x.columns
            y_feature = self.y.columns

            # Joined and prepare the whole dataframe
            joined_dataframe = pd.concat([self.x, self.y], axis=1)
            # Dropped all extra null values
            joined_dataframe.dropna(inplace=True)
            # Retrieved X and y again
            self.x = joined_dataframe[x_features]
            self.y = joined_dataframe[y_feature]

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
        print(f"Initiating Preprocessing...")
        # Using Null Processor
        print(f"Going through null values and features...")
        null_dropper = NullProcessing(self.x, self.y)
        self.x, self.y = null_dropper.run()
        print(f"Null values and features processed...")

        print(f"Finishing Preprocessing...")

        return self.x, self.y
