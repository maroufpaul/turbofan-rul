import numpy as np

def phm08_score(y_true, y_pred):
    """
    Piecewise-exponential penalty used in the PHM08 competition.
    Positive error (late prediction) is penalised harder.
    """
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return np.mean(score)

def regression_metrics(y_true, y_pred):
    mae  = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    phm  = phm08_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "PHM08": phm}
