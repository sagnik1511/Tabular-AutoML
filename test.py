"""
This file is for testing the integration of the library classes and functions.
"""
from tab_automl.automl.Datasets import Iris, Wine
from tab_automl.automl.training import Trainer
from tab_automl.automl.processing import PreProcessing
from tab_automl.automl.fet_engineering import FeatureEngineering
from tab_automl.utils.training import train_validation_split


def classification_test():
    print("Testing through Classification AutoML ...")
    # Loading the dataset
    dataset = Iris()
    # X feature set and target feature split
    x, y = dataset.prepare_x_and_y()
    # Defining processor
    processor = PreProcessing(x=x, y=y)
    # Executing processor
    processor.run()
    # Saving processed data
    processor.save_data()
    # Defining feature engineer
    feature_engineer = FeatureEngineering(x=x, y=y)
    # Executing feature engineer
    feature_engineer.run()
    # Saving engineered data
    feature_engineer.save_data()
    # Defining model trainer
    trainer = Trainer(problem_type="classification")
    # Preparing train and validation split
    x_train, y_train, x_val, y_val = train_validation_split(x=x, y=y)
    # Training AutoML and saving the best model
    trainer.single_model_trainer(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, save_model=True)
    print("Classification test completed successfully...\n")


def regression_test():
    print("Testing through Regression AutoML ...")
    # Loading the dataset
    dataset = Wine()
    # X feature set and target feature split
    x, y = dataset.prepare_x_and_y()
    # Defining processor
    processor = PreProcessing(x=x, y=y)
    # Executing processor
    processor.run()
    # Saving processed data
    processor.save_data()
    # Defining feature engineer
    feature_engineer = FeatureEngineering(x=x, y=y)
    # Executing feature engineer
    feature_engineer.run()
    # Saving engineered data
    feature_engineer.save_data()
    # Defining model trainer
    trainer = Trainer(problem_type="regression")
    # Preparing train and validation split
    x_train, y_train, x_val, y_val = train_validation_split(x=x, y=y)
    # Training AutoML and saving the best model
    trainer.single_model_trainer(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, save_model=True)
    print("Regression test completed successfully...\n")


def test():
    # Testing classification
    classification_test()
    # Testing Regression
    regression_test()


if __name__ == "__main__":
    test()
