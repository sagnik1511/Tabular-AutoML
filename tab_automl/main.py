"""
This is the main executable file
"""
import argparse
from tab_automl.automl import datasets, processing, fet_engineering, training
from tab_automl.utils.misc import validate_parser_variable
from tab_automl.utils.training import train_validation_split

# Defining parser
parser = argparse.ArgumentParser(description="automl hyper parameters")
parser.add_argument("-d", "--data-source", type=str, required=True, metavar="", help="File path")
parser.add_argument("-t", "--problem-type", type=str, required=True, metavar="",
                    help="Problem Type , currently supporting *regression* or *classification*")
parser.add_argument("-tf", "--target-feature", type=str, required=True, metavar="",
                    help="Target feature inside the data")
parser.add_argument("-p", "--pre-proc", type=str, default="true", metavar="",
                    help="If data processing is required")
parser.add_argument("-f", "--fet-eng", type=str, default="true", metavar="",
                    help="If feature engineering is required")
parser.add_argument("-spd", "--save-proc-data", type=str, default="true", metavar="",
                    help="Save the processed data")
parser.add_argument("-sfd", "--save-fet-data", type=str, default="true", metavar="",
                    help="Save the feature engineered data")
parser.add_argument("-sm", "--save-model", type=str, default="true", metavar="",
                    help="Save the best trained model")

# Main function
if __name__ == "__main__":
    # Retrieving parser variables
    args = parser.parse_args()
    print("Parser data collected...")
    # Validating parser variables
    print(args)
    validate_parser_variable(args)
    print("Parser variables validated successfully...")
    # Feeding the data to the class of respective problem statement.
    if args.problem_type == "classification":
        dataset = datasets.ClassificationDataset(args.data_source)
    else:
        dataset = datasets.RegressionDataset(args.data_source)
    print("Populated the dataframe with data records...")

    # Accessing the feature names to validate the X and y of the data
    features = dataset.data.columns.tolist()
    assert args.target_feature in features, "Target feature not found in Dataset!"
    # Omitting the target variable from the X feature set columns
    features.remove(args.target_feature)
    x_features = features
    print(f"Features taken for X : {x_features}")
    print(f"Target feature : {args.target_feature}")
    # Splitting the data into X and y
    X, y = dataset.prepare_x_and_y(feature_set_columns=x_features,
                                   target_column=args.target_feature)
    print(f"X feature set and target feature split...")

    if args.pre_proc == "true":
        # Defining data processing class
        processor = processing.PreProcessing(X, y)
        # processing data
        X, y = processor.run()
        # Saving the processed data if required
        if args.save_proc_data == "true":
            processor.save_data()

    if args.fet_eng == "true":
        # Defining the feature engineering class
        feature_engineer = fet_engineering.FeatureEngineering(X, y)
        # engineering features
        X, y = feature_engineer.run()
        # Saving the feature engineered data if required
        if args.save_fet_data == "true":
            feature_engineer.save_data()

    # Creating a validation data split from training data
    X_train, y_train, X_val, y_val, val_size = train_validation_split(X, y)
    print(f"Validation data prepared."
          f" Train- Validation ratio taken {int(100 - val_size * 100)} % - {int(val_size * 100)} %")

    # Defining trainer
    trainer = training.Trainer(problem_type=args.problem_type)
    # Training models on the data
    save_model = args.save_model == "true"  # Defining the model saving
    trainer.single_model_trainer(X_train, y_train, X_val, y_val, save_model=save_model)

    print("AutoML executed successfully...")
