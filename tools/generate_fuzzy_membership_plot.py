"""
Skrypt do wizualizacji funkcji przynależności regulatora Fuzzy-LQR.

Generuje wykres z czterema podwykresami pokazującymi trójkątne funkcje
przynależności dla każdej zmiennej stanu: theta, theta_dot, x, x_dot.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Parametry funkcji przynależności z fuzzy_lqr.py (zoptymalizowane)
TH_SMALL = (-0.15, 0.0, 0.15)     # theta [rad]
THD_SMALL = (-1.0, 0.0, 1.0)     # theta_dot [rad/s]
X_SMALL = (-0.3, 0.0, 0.3)       # x [m]
XD_SMALL = (-0.5, 0.0, 0.5)      # x_dot [m/s]


def tri_mf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Trójkątna funkcja przynależności - wersja wektoryzowana."""
    result = np.zeros_like(x)
    
    # Lewa rampa: (x - a) / (b - a)
    left_mask = (x > a) & (x <= b)
    result[left_mask] = (x[left_mask] - a) / (b - a + 1e-12)
    
    # Prawa rampa: (c - x) / (c - b)
    right_mask = (x > b) & (x < c)
    result[right_mask] = (c - x[right_mask]) / (c - b + 1e-12)
    
    return result


def plot_membership_functions():
    """Generuje wykres funkcji przynależności dla wszystkich zmiennych stanu."""
    
    # Konfiguracja wykresu (bez tytułu - opis w LaTeX)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    
    # Kolory
    color_small = '#2E86AB'  # niebieski
    color_large = '#A23B72'  # różowy/magenta
    
    # Lista zmiennych: (ax_idx, params, label, unit, x_range_mult)
    variables = [
        ((0, 0), TH_SMALL, r'$\theta$', 'rad', 2.5),
        ((0, 1), THD_SMALL, r'$\dot{\theta}$', 'rad/s', 2.0),
        ((1, 0), X_SMALL, r'$x$', 'm', 2.5),
        ((1, 1), XD_SMALL, r'$\dot{x}$', 'm/s', 2.0),
    ]
    
    for (i, j), params, var_label, unit, range_mult in variables:
        ax = axes[i, j]
        a, b, c = params
        
        # Zakres osi x
        x_min = a * range_mult
        x_max = c * range_mult
        x = np.linspace(x_min, x_max, 500)
        
        # Funkcje przynależności
        mu_small = tri_mf(x, a, b, c)
        mu_large = 1.0 - mu_small
        
        # Rysowanie
        ax.plot(x, mu_small, color=color_small, linewidth=2.5, label='Mały błąd')
        ax.plot(x, mu_large, color=color_large, linewidth=2.5, label='Duży błąd')
        ax.fill_between(x, mu_small, alpha=0.2, color=color_small)
        ax.fill_between(x, mu_large, alpha=0.2, color=color_large)
        
        # Pionowe linie dla punktów charakterystycznych
        for val, style in [(a, '--'), (b, ':'), (c, '--')]:
            ax.axvline(val, color='gray', linestyle=style, linewidth=0.8, alpha=0.7)
        
        # Etykiety - powiększone
        ax.set_xlabel(f'{var_label} [{unit}]', fontsize=14)
        ax.set_ylabel(r'$\mu$', fontsize=14)
        
        # Adnotacje z wartościami granicznymi - powiększone
        ax.annotate(f'{a:.2f}', xy=(a, 0), xytext=(a, -0.12),
                   ha='center', fontsize=11, color='gray')
        ax.annotate(f'{c:.2f}', xy=(c, 0), xytext=(c, -0.12),
                   ha='center', fontsize=11, color='gray')
        
        # Siatka i zakres
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.3, linestyle='-')
        ax.legend(loc='upper right', fontsize=12)
        ax.tick_params(axis='both', labelsize=12)
        
        # Tło
        ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    
    # Zapis do pliku
    output_dir = Path(__file__).parent.parent / 'latex' / 'images' / 'diagrams'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'fuzzy_membership.png'
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Zapisano wykres: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    plot_membership_functions()

