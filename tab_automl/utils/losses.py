from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_squared_log_error


loss_fn_dict = {
    "accuracy_score": [accuracy_score, "+"],
    "mse": [mean_squared_error, "-"],
    "msle": [mean_squared_log_error, "-"],
    "f1_score": [f1_score, "+"]
}


def calc_metric(x, y, model, metric):
    pred = model.predict(x)
    return metric(y, pred)


def fetch_metric_scores(train_set, val_set, trained_model, metrics=None):
    metric_dict = dict()
    x_train, y_train = train_set
    x_val, y_val = val_set
    if y_train.shape[1] > 1:
        y_train = y_train.iloc[:, 0]
    if y_val.shape[1] > 1:
        y_val = y_val.iloc[:, 0]

    for metric_name in metrics:
        print(f"Scoring on {metric_name}")
        metric, metric_type = loss_fn_dict[metric_name]
        train_metric_score = calc_metric(x_train, y_train, trained_model, metric)
        val_metric_score = calc_metric(x_val, y_val, trained_model, metric)
        print(f"train set score : {train_metric_score}   ||  "
              f"validation set score : {val_metric_score}")
        metric_dict[f"train_{metric_name}"] = [train_metric_score, metric_type]
        metric_dict[f"val_{metric_name}"] = [val_metric_score, metric_type]

    return metric_dict
