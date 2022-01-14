"""
This file will hold miscellaneous utilities.
"""
import os


def check_true_false(variable):
    """
    Checks if the string variable contains
    "true" or "false" else raises error

    Args:
        variable: (str) input variable to be checked

    """
    if variable == "true" or variable == "false":
        return True
    else:
        raise TypeError


def validate_parser_variable(args):
    """
    Validates the parser variables if in correct state.

    Args:
        args: (argparse.Namespace) arguments parsed by main file

    """

    assert os.path.isfile(args.data_source) == True, "Invalid data source, data source is not found..."
    assert args.problem_type == "classification" or args.problem_type == "regression", "Problem Type not found..."
    assert check_true_false(args.pre_proc), "Variable must be named true or false"
    assert check_true_false(args.fet_eng), "Variable must be named true or false"
    assert check_true_false(args.save_proc_data), "Variable must be named true or false"
    assert check_true_false(args.save_fet_data), "Variable must be named true or false"
    assert check_true_false(args.save_model), "Variable must be named true or false"
