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
    
    # Ustawienia globalne dla większych czcionek (prezentacja)
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
    })
    
    # Konfiguracja wykresu - większy rozmiar dla prezentacji
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
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
        
        # Rysowanie - grubsze linie
        ax.plot(x, mu_small, color=color_small, linewidth=3.5, label='Mały błąd')
        ax.plot(x, mu_large, color=color_large, linewidth=3.5, label='Duży błąd')
        ax.fill_between(x, mu_small, alpha=0.25, color=color_small)
        ax.fill_between(x, mu_large, alpha=0.25, color=color_large)
        
        # Pionowe linie dla punktów charakterystycznych
        for val, style in [(a, '--'), (b, ':'), (c, '--')]:
            ax.axvline(val, color='gray', linestyle=style, linewidth=1.2, alpha=0.7)
        
        # Etykiety - duże czcionki dla prezentacji
        ax.set_xlabel(f'{var_label} [{unit}]', fontsize=22)
        ax.set_ylabel(r'$\mu$', fontsize=22)
        
        # Adnotacje z wartościami granicznymi - przeniesione na górę
        ax.annotate(f'{a:.2f}', xy=(a, 1.0), xytext=(a, 1.08),
                   ha='center', fontsize=14, color='#555555')
        ax.annotate(f'{c:.2f}', xy=(c, 1.0), xytext=(c, 1.08),
                   ha='center', fontsize=14, color='#555555')
        
        # Siatka i zakres
        ax.set_ylim(-0.02, 1.18)
        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.3, linestyle='-')
        ax.legend(loc='upper right', fontsize=18, framealpha=0.9)
        ax.tick_params(axis='both', labelsize=16, width=1.5, length=6)
        
        # Grubsze ramki
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        # Tło
        ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout(pad=2.0)
    
    # Zapis do pliku - wyższa rozdzielczość
    output_dir = Path(__file__).parent.parent / 'latex' / 'images' / 'diagrams'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'fuzzy_membership.png'
    
    plt.savefig(output_path, dpi=250, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Zapisano wykres: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    plot_membership_functions()

