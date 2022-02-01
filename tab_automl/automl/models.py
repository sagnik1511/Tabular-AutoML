"""
This files holds all codes for defined and used models through the life cycle of the package.
"""
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.svm import SVR 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor


single_model_dict = {
    "regression": {
        "Linear Regression": LinearRegression,
        "Lasso Regression": Lasso,
        "Ridge Regression": Ridge,
        "Random Forest Regression": RandomForestRegressor,
        "Support Vector Regression": SVR
        "KNN Regressor": KNeighborsRegressor
    },
    "classification": {
        "Decision Tree Classifier": DecisionTreeClassifier,
        "Light Gradient Boosting Classifier": LGBMClassifier,
        "Random Forest Classifier": RandomForestClassifier,
        "XGBoost Classifier": XGBClassifier
        "KNN classifier":KNeighborsClassifier
    }
}
