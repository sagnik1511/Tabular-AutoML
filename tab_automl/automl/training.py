"""
This file holds all codes for training models of processed feature engineered data.
"""
import pickle
import os
import numpy as np
from tab_automl.automl.models import single_model_dict
from tab_automl.utils.losses import fetch_metric_scores
from tab_automl.utils.losses import loss_fn_dict
from termcolor import cprint
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
                             metric_list=["accuracy_score"],
                             model_dict=single_model_dict,
                             result_monitor="accuracy_score",
                             check_on="val",
                             save_model=True):
        """
        Trains models of selected problem type and find the best results

        Args:
            x_train: (pandas.DataFrame) Feature set / Affecting features for training
            y_train: (pandas.Dataframe) Target set / dependent feature for training
            x_val: (pandas.DataFrame) Feature set / Affecting features for validation
            y_val: (pandas.Dataframe) Target set / dependent feature for validation
            metric_list: (List[Any]) List of metric on which the model will be tested
            model_dict: (Any) Model zoo for problem type
            result_monitor: (Any) Model metric to monitor best results
            check_on: (Any) The responsible dataset for model update
            save_model: (bool) whether the best model will be saved

        """
        # Checking the data shape whether iit has same number of records
        assert x_train.shape[0] == y_train.shape[0] or x_val.shape[0] == y_val.shape[0], "Data shape mismatched..."
        assert result_monitor in metric_list, "metric not found in metric list..."
        print(f"Problem statement selected : {self.problem_type} .")
        print(f"Best model validator : {check_on}_{result_monitor}")
        # Selecting required models
        models = model_dict[self.problem_type]
        # Model Training
        print(f"Initiating Model Training...")
        # Declaring the best model
        self.result_monitor = result_monitor
        self.check_on = check_on
        self.best_score = np.inf if loss_fn_dict[result_monitor][1] == "-" else 0
        self.is_loss = True if loss_fn_dict[result_monitor][1] == "-" else False
        self.best_model = None
        self.best_model_name = None
        # Iterating through models
        for model_name in models.keys():
            cprint(f"\n{model_name} taken for training...", "blue")
            # Declaring the model
            model = models[model_name]()
            start_time = time.time()
            # Training the model
            model.fit(x_train, y_train)
            # fetching metric_scores
            metric_scores = fetch_metric_scores((x_train, y_train),
                                                (x_val, y_val), metrics=metric_list,
                                                trained_model=model)
            print(f"Model Metrics :")
            print({k: '%.5f'%v[0] for k, v in metric_scores.items()})
            self.model_checkpoint(metric_scores, model_name, model)

        print(f"Model training completed...")
        # Saving the best model
        if save_model:
            with open("best_model.pkl", "wb") as outfile:
                pickle.dump(self.best_model, outfile)
                outfile.close()
            # Validating output path
            output_path = os.path.join(os.getcwd(), "best_model.pkl")
            assert os.path.isfile(output_path), "Model saved on different location!"
            print(f"Best model saved at {output_path}")

    def model_checkpoint(self, metric_dict, model_name, model):
        flag = True
        if self.is_loss:
            if self.best_score > metric_dict[f"{self.check_on}_{self.result_monitor}"][0]:
                cprint("Model Updated...", "green")
                cprint(f"Current best model : {model_name}", "green")
                self.best_score = metric_dict[f"{self.check_on}_{self.result_monitor}"][0]
                self.best_model_name = model_name
                self.best_model = model
                cprint(f"Current best {self.check_on}_{self.result_monitor} "
                       f"value : {metric_dict[f'{self.check_on}_{self.result_monitor}'][0]}", "cyan")
                flag = False
        else:
            if self.best_score < metric_dict[f"{self.check_on}_{self.result_monitor}"][0]:
                cprint(f"Model Updated...", "green")
                cprint(f"Current best model : {model_name}", "green")
                self.best_score = metric_dict[f"{self.check_on}_{self.result_monitor}"][0]
                self.best_model_name = model_name
                self.best_model = model
                cprint(f"Current best {self.check_on}_{self.result_monitor} "
                      f"value : {metric_dict[f'{self.check_on}_{self.result_monitor}'][0]}", "cyan")
                flag = False

        if flag:
            cprint(f"Model update failed...", "red")
