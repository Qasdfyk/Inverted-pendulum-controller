
import numpy as np

def mse(y, yref):
    y, yref = np.asarray(y), np.asarray(yref)
    return float(np.mean((y - yref)**2))

def mae(y, yref):
    y, yref = np.asarray(y), np.asarray(yref)
    return float(np.mean(np.abs(y - yref)))

def print_summary(metrics: dict):
    print(f"MSE(theta)={metrics['mse_theta']:.6f}  MAE(theta)={metrics['mae_theta']:.6f}  "
          f"MSE(x)={metrics['mse_x']:.6f}  MAE(x)={metrics['mae_x']:.6f}")
