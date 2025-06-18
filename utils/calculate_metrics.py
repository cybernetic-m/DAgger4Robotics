from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true_list, y_pred_list, metrics):

    rmse = root_mean_squared_error(y_true=y_true_list, y_pred=y_pred_list)
    mae = mean_absolute_error(y_true=y_true_list, y_pred=y_pred_list)
    r2 = r2_score(y_true=y_true_list, y_pred=y_pred_list)

    metrics['rmse'].append(rmse)
    metrics['mae'].append(mae)
    metrics['r2'].append(r2)

    return rmse, mae, r2