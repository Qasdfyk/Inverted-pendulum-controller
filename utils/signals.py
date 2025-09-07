
import numpy as np

def position_step(t, x_ref=0.1):
    return x_ref

def angle_step(t, theta_ref=0.0):
    return theta_ref

def track_sine(t, amp=0.05, w=1.0):
    return amp*np.sin(w*t)

def impulse(t, t0=0.5, A=0.05):
    return A if abs(t - t0) < 1e-3 else 0.0
