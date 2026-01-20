import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import controllers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from controllers.mpc_utils import Wind, SIM

# Output path
SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../latex/img/wind_signal.png'))

# Matplotlib configuration for large fonts
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'lines.linewidth': 3
})

def main():
    # Parameters from run_experiments.py
    T = SIM["T"] # 10.0
    dt = SIM["dt"] # 0.1
    
    # Create Wind instance using the same seed as experiments
    # run_experiments.py line 90: wind = Wind(T, seed=23341, power=1e-3, smooth=5)
    wind = Wind(T, seed=23341, power=1e-3, smooth=5)
    
    # Generate time grid for plotting
    t = np.arange(0, T + dt, dt)
    
    # Generate wind signal
    # Wind class is callable
    wind_signal = [wind(ti) for ti in t]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t*10, wind_signal, label='Wiatr', linewidth=3)
    ax.set_ylabel(r'$F_w$[N]')
    ax.set_xlabel('k')
    ax.grid(True)
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    plt.savefig(SAVE_PATH)
    print(f"Plot saved to {SAVE_PATH}")
    plt.close(fig)

if __name__ == "__main__":
    main()
