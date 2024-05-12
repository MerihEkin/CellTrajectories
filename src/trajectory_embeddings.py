import numpy as np
import pandas as pd

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


def interpolate_tracjectory(x, y, t : np.ndarray):
    """
    Implements linear interpolation to x, y
    returns: interpolated valued of x, y, and new t array
    """
    t_new = np.arange(t.min(), t.max(), step=1)
    x_interp = interp1d(t, x, kind='linear')    # Linear interpolation is sufficient
    y_interp = interp1d(t, y, kind='linear') 
    return x_interp(t_new), y_interp(t_new), t_new


def filter_trajectory(x : np.ndarray, y : np.ndarray, window_length = 15):
    """
    smooths x and y array using savitzky-golay filters
    """
    x_smooth = savgol_filter(x, window_length=window_length, polyorder=2)
    y_smooth = savgol_filter(y, window_length=window_length, polyorder=2)
    return x_smooth, y_smooth


def get_embedding(values : np.ndarray):
    """
    return minium, maximum, mean, 25th, 50th (median), 75th percentile of value 
    """
    return np.stack([np.min(values), 
              np.max(values),
              np.mean(values),
              np.percentile(values, 25),
              np.percentile(values, 50),
              np.percentile(values, 75),
              ])


def trajectory_embeddings(track : pd.DataFrame, dt, window_size):
    """"
    Takes trajectory dataframe and returns data with embeddings
    Note : it clips the data with window_size//2 from start and 
    end to avoid generating bias in the lstn-encoder. 

    returns : dataframe
    """
    x = track['x'].to_numpy()
    y = track['y'].to_numpy()
    t = track['t'].to_numpy()
    x_interp, y_interp, t_new = interpolate_tracjectory(x, y, t)
    x_smooth, y_smooth = filter_trajectory(x_interp, y_interp)

    trajectory = np.stack((x_smooth, y_smooth), axis=-1)
    vel = np.diff(trajectory, axis=0) / dt
    vel_norm = np.linalg.norm(vel, axis=1)
    acc = np.diff(vel_norm)
    cos_angle = np.einsum('ij,ij->i', vel[:-1], vel[1:]) / (vel_norm[:-1] * vel_norm[1:])
    omega = np.arccos(np.clip(cos_angle, -1, 1))

    step = window_size // 2
    indices = np.arange(step, len(acc) - step)
    
    v_embeddings = [get_embedding(vel_norm[i-step:i+step]) for i in indices]
    a_embeddings = [get_embedding(acc[i-step:i+step]) for i in indices]
    w_embeddings = [get_embedding(omega[i-step:i+step]) for i in indices]

    data = {
        'x': x_smooth[indices],
        'y': y_smooth[indices],
        't': t_new[indices],
        'v': v_embeddings,
        'a': a_embeddings,
        'w': w_embeddings
    }

    return pd.DataFrame(data)






