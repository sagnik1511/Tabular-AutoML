"""
This files holds all codes for defined and used models through the life cycle of the package.
"""
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier


single_model_dict = {
    "regression": {
        "Linear Regression": LinearRegression,
        "Lasso Regression": Lasso,
        "Ridge Regression": Ridge,
        "Random Forest Regression": RandomForestRegressor,
        "Support Vector Regression": SVR
    },
    "classification": {
        "Decision Tree Classifier": DecisionTreeClassifier,
        "Light Gradient Boosting Classifier": LGBMClassifier,
        "Random Forest Classifier": RandomForestClassifier,
        "XGBoost Classifier": XGBClassifier
    }
}
