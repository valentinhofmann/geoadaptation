import numpy as np


def evaluate(c1, c2, scale_km=True):
    d = np.radians(c2-c1)
    a = np.sin(d[:, 0]/2) * np.sin(d[:, 0]/2) + np.cos(np.radians(c1[:, 0])) * np.cos(np.radians(c2[:, 0])) * np.sin(d[:, 1]/2) * np.sin(d[:, 1]/2)
    d = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    if scale_km:
        return 6371 * d
    else:
        return d


def mean_dist(points_pred, points_true, scaler, geo_type):
    if geo_type == 'kmeans':
        points_pred = np.array(points_pred)
        points_true = scaler.inverse_transform(points_true)
    else:
        points_pred = scaler.inverse_transform(points_pred)
        points_true = scaler.inverse_transform(points_true)
    d = evaluate(points_pred, points_true)
    return np.mean(d)


def median_dist(points_pred, points_true, scaler, geo_type):
    if geo_type == 'kmeans':
        points_pred = np.array(points_pred)
        points_true = scaler.inverse_transform(points_true)
    else:
        points_pred = scaler.inverse_transform(points_pred)
        points_true = scaler.inverse_transform(points_true)
    d = evaluate(points_pred, points_true)
    return np.median(d)
