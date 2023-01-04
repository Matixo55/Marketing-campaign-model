import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from pipelines.boolean_features import replace_obscured_booleans
from pipelines.categorical_features import encode_categorical_columns
from pipelines.generic import drop_columns
from pipelines.nulls_processing import (
    detect_hidden_nulls,
    replace_hidden_nulls,
    replace_nulls_in_numeric_columns,
    replace_nulls_in_categorical_columns,
)
from pipelines.numeric_features import (
    winsortize_outliers_in_columns,
    standardize_numeric_columns,
)

pd.set_option("display.max_columns", None)


def objective(trial):
    classifier_name = trial.suggest_categorical(
        "classifier",
        [
            "SVC",  #
            # "KNeighborsClassifier",
            # "LogisticRegression",
            # "GaussianNB",
            # "DecisionTreeClassifier",
            # "RandomForestClassifier",
            # "XGBClassifier",
        ],
    )

    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])

        classifier_obj = SVC(C=svc_c, gamma="auto", kernel=kernel)
    elif classifier_name == "KNeighborsClassifier":
        n_neighbors = trial.suggest_int("n_neighbors", 1, 50)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])

        classifier_obj = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    elif classifier_name == "LogisticRegression":
        C = trial.suggest_float("C", 1e-10, 1e10, log=True)

        classifier_obj = LogisticRegression(C=C)
    elif classifier_name == "GaussianNB":
        classifier_obj = GaussianNB()
    elif classifier_name == "DecisionTreeClassifier":
        max_depth = trial.suggest_int("max_depth", 1, 6)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

        classifier_obj = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    elif classifier_name == "RandomForestClassifier":
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 6)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

        classifier_obj = RandomForestClassifier(max_depth=rf_max_depth, criterion=criterion)
    else:
        max_depth = trial.suggest_int("max_depth", 1, 6)
        learning_rate = trial.suggest_float("learning_rate", 1e-10, 1e10, log=True)

        classifier_obj = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate)

    score = cross_val_score(classifier_obj, X_train, y_train, n_jobs=-1, cv=3, scoring="roc_auc")

    return score.mean()


if __name__ == "__main__":
    data = pd.read_csv("data/raw_data.csv", sep=";")
    boolean_columns = ["default", "housing", "loan"]

    data = replace_obscured_booleans(data, boolean_columns + ["y"])
    data = replace_hidden_nulls(data, data.columns.tolist())

    X = data.drop(columns=["y"])
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # detect_hidden_nulls(data, data.columns)

    X_train = drop_columns(
        X_train,
        [
            "euribor3m",  # high correlation with cons.conf.idx
            "emp.var.rate",  # high correlation with emp.var.rate
            "duration",  # is known only after the call
            "month",  # unnatural correlation with result, also won't work if next campaign is in another month
            "pdays",  # 96% of values are nulls
            "poutcome",  # 86% of values are nulls
        ],
    )
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns.values.tolist()
    categorical_columns = X_train.select_dtypes(exclude=[np.number]).columns.values.tolist()

    for column in boolean_columns:
        numeric_columns.remove(column)

    X_train, X_test = replace_nulls_in_numeric_columns(
        X_train, X_test, numeric_columns, strategy="mean"
    )
    X_train, X_test = replace_nulls_in_categorical_columns(
        X_train, X_test, categorical_columns + boolean_columns, strategy="most_frequent"
    )

    categorical_columns.remove("day_of_week")
    categorical_columns.remove("education")

    X_train, X_test = encode_categorical_columns(
        X_train,
        X_test,
        "education",
        strategy="ordinal",
        categories=[
            "illiterate",
            "basic.4y",
            "basic.6y",
            "basic.9y",
            "high.school",
            "professional.course",
            "university.degree",
        ],
    )
    X_train, X_test = encode_categorical_columns(
        X_train,
        X_test,
        "day_of_week",
        strategy="ordinal",
        categories=["mon", "tue", "wed", "thu", "fri"],
    )
    X_train, X_test = encode_categorical_columns(
        X_train, X_test, ["day_of_week"], strategy="trigonometric"
    )

    X_train, X_test = encode_categorical_columns(
        X_train, X_test, categorical_columns, strategy="onehot"
    )
    X_train, X_test = winsortize_outliers_in_columns(
        X_train,
        X_test,
        numeric_columns,
        capping_method="gaussian",
    )
    X_train, X_test = standardize_numeric_columns(X_train, X_test, numeric_columns + ["education"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print("xdxdxdxd", study.best_trial)
