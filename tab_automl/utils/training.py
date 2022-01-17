"""
This file holds all utilities for training.
"""
from sklearn.model_selection import train_test_split


def train_validation_split(x, y):
    """
    Prepare validation data with proper size

    Args:
        x: (pandas.DataFrame) Feature set / Affecting features
        y: (pandas.Dataframe) Target set / dependent feature

    Returns:
        x_train: (pandas.DataFrame) Feature set / Affecting features for training
        y_train: (pandas.Dataframe) Target set / dependent feature for training
        x_val: (pandas.DataFrame) Feature set / Affecting features for validation
        y_val: (pandas.Dataframe) Target set / dependent feature for validation

    """

    # For large datasets
    if x.shape[0] > 100000:
        val_ratio = 0.2
    # For medium size datasets
    elif x.shape[0] > 1000:
        val_ratio = 0.15
    # For small datasets
    else:
        val_ratio = 0.1

    # Splitting dataset into train and validation
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_ratio, random_state=42)
    print(f"Validation data prepared."
          f" Train - Validation ratio taken {int(100 - val_ratio * 100)} % - {int(val_ratio * 100)} % .")

    return x_train, y_train, x_val, y_val
