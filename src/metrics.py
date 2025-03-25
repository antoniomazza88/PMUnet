import numpy as np
import json

def PSNR(pred, true, maxval=1):
    mse = MSE(pred, true)
    psnr = -10 * np.log10(mse / (maxval ** 2))
    return psnr.mean()


def MSE(pred, true):
    mse = ((pred - true) ** 2)
    mse = np.mean(mse)
    return mse


def RMSE(pred, true):
    mse = ((pred - true) ** 2)
    mse = np.mean(mse)
    return np.sqrt(mse)

def eval_perf(o, p):
    perf = dict()
    perf['MAE'] = float(np.mean(np.abs(o - p)))
    perf['RMSE'] = float(RMSE(o, p))
    perf['PSNR'] = float(PSNR(o, p, 1))
    return perf


def eval_overall(perf):
    dates = perf.keys()
    dates = list(dates)
    dates = [x for x in dates if not x.__contains__('overall')]
    metrics = perf[list(dates)[0]].keys()
    perf['overall'] = dict()
    for m in metrics:
        metr = 0
        for d in dates:
            metr += perf[d][m]
            perf['overall'][m] = metr / len(dates)
    return perf