"""
This file holds all codes for training models of processed feature engineered data.
"""
import pickle
import os
from tab_automl.automl.models import single_model_dict
import time
import warnings
warnings.filterwarnings("ignore")


class Trainer:
    """

    Trainer class to train models
    and find the best model and tuning

    """
    def __init__(self, problem_type):
        """

        Args:
            problem_type: (string) The problem statement to find right models

        """
        self.problem_type = problem_type

    def single_model_trainer(self, x_train, y_train,
                             x_val, y_val,
                             model_dict=single_model_dict,
                             result_monitor="val_score",
                             save_model=True):
        """
        Trains models of selected problem type and find the best results

        Args:
            x_train: (pandas.DataFrame) Feature set / Affecting features for training
            y_train: (pandas.Dataframe) Target set / dependent feature for training
            x_val: (pandas.DataFrame) Feature set / Affecting features for validation
            y_val: (pandas.Dataframe) Target set / dependent feature for validation
            model_dict: (Any) Model zoo for problem type
            result_monitor: (Any) Model metric to monitor best results
            save_model: (bool) whether the best model will be saved

        """
        # Checking the data shape whether iit has same number of records
        assert x_train.shape[0] == y_train.shape[0] or x_val.shape[0] == y_val.shape[0], "Data shape mismatched..."

        print(f"Problem statement selected : {self.problem_type} .")
        # Selecting required models
        models = model_dict[self.problem_type]
        # Model Training
        print(f"Initiating Model Training...")
        # Declaring best scores to save the best model
        best_score = 0
        best_model = None
        best_model_name = None
        # Iterating through models
        for model_name in models.keys():
            print(f"{model_name} taken for training...")
            # Declaring the model
            model = models[model_name]()
            start_time = time.time()
            # Training the model
            model.fit(x_train, y_train)
            # Storing accuracy scores
            train_score = model.score(x_train, y_train)
            val_score = model.score(x_val, y_val)
            print(f"[ {model_name} ] Model trained in {'%.4f'%(time.time() - start_time)} seconds")
            print(f"[ {model_name} ] Model Accuracy Score on training data : {'%.6f'%train_score}")
            print(f"[ {model_name} ] Model Accuracy Score on validation data : {'%.6f'%val_score}")
            # Model Checkpoint to save the best result and model
            if result_monitor == "val_score":
                # Using validation score to monitor best model
                if val_score > best_score:
                    best_model = model
                    best_model_name = model_name
                    best_score = val_score
                    print(f"Results upgraded. "
                          f"Current Best Model : {best_model_name}. "
                          f"Current best validation score : {'%.6f'%best_score}")
                else:
                    print(f"Results didn't upgraded. "
                          f"Current Best Model : {best_model_name}. "
                          f"Current best validation score : {'%.6f'%best_score}")
            else:
                # Using train score to monitor best model
                if train_score > best_score:
                    best_model = model
                    best_model_name = model_name
                    best_score = train_score
                    print(f"Results upgraded."
                          f"Current Best Model : {best_model_name}."
                          f"Current best train score : {'%.6f'%best_score}")
                else:
                    print(f"Results didn't upgraded."
                          f"Current Best Model : {best_model_name}."
                          f"Current best validation score : {'%.6f' % best_score}")

        print(f"Model training completed...")

        # Saving the best model
        if save_model:
            with open("best_model.pkl", "wb") as outfile:
                pickle.dump(best_model, outfile)
                outfile.close()
            # Validating output path
            output_path = os.path.join(os.getcwd(), "best_model.pkl")
            assert os.path.isfile(output_path), "Model saved on different location!"
            print(f"Best model saved at {output_path}")
