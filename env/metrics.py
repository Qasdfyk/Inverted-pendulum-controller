import numpy as np

# ====== Błędy bazowe (zostawiam Twoje) ======
def mse(y, yref):
    y, yref = np.asarray(y), np.asarray(yref)
    return float(np.mean((y - yref)**2))

def mae(y, yref):
    y, yref = np.asarray(y), np.asarray(yref)
    return float(np.mean(np.abs(y - yref)))

# ====== Nowe: całki z błędów ======
def iae(t, y, yref):
    """Integral of Absolute Error"""
    t = np.asarray(t); e = np.abs(np.asarray(y) - np.asarray(yref))
    return float(np.trapz(e, t))

def ise(t, y, yref):
    """Integral of Squared Error"""
    t = np.asarray(t); e = np.asarray(y) - np.asarray(yref)
    return float(np.trapz(e*e, t))

# ====== Energia sterowania / zużycie energii ======
def control_energy_l2(t, u):
    """∫ u(t)^2 dt — 'energia sterowania' (L2)"""
    t = np.asarray(t); u = np.asarray(u)
    return float(np.trapz(u*u, t))

def control_energy_l1(t, u):
    """∫ |u(t)| dt — przybliżenie zużycia energii/‘paliwa’ aktuatora (L1)"""
    t = np.asarray(t); u = np.asarray(u)
    return float(np.trapz(np.abs(u), t))

# ====== Czas ustalania (z histerezą utrzymania) ======
def settling_time(t, y, yref, eps, hold_time):
    """
    Najwcześniejszy czas, od którego |y - yref| <= eps utrzymuje się
    nieprzerwanie >= hold_time. Jeśli brak — zwraca np.nan.
    """
    t = np.asarray(t); e = np.abs(np.asarray(y) - np.asarray(yref))
    inside = e <= eps
    # Szukamy najwcześniejszego indeksu, od którego do końca okna hold_time wszystko jest 'inside'
    N = len(t)
    if N < 2: return float('nan')
    dt = np.diff(t).mean()  # równomierna siatka w tym projekcie
    win = max(1, int(np.round(hold_time / dt)))
    # Dla każdego i sprawdź, czy wszystkie w [i, i+win) są True
    for i in range(N - win + 1):
        if np.all(inside[i:i+win]):
            return float(t[i])
    return float('nan')

# ====== Overshoot ======
def overshoot(y, yref_final):
    """
    Maksymalne przeregulowanie względem końcowej wartości referencyjnej.
    Zwraca procent (0..), lub NaN gdy ref_final ~ 0 (niezdefiniowane).
    """
    y = np.asarray(y); r = float(yref_final)
    if np.isclose(r, 0.0, atol=1e-12):
        return float('nan')
    peak = float(np.max(y)) if r >= y[0] else float(np.min(y))
    return float(100.0 * (peak - r) / abs(r))

# ====== Błąd ustalony ======
def steady_state_error(t, y, yref, window_frac=0.1, window_min=0.5):
    """
    Średni błąd na końcówce (ostatnie window_frac czasu, min window_min sek).
    Zwraca średnią i RMS dla wglądu.
    """
    t = np.asarray(t); y = np.asarray(y); yref = np.asarray(yref)
    T = t[-1] - t[0] if len(t) > 1 else 0.0
    w = max(window_min, window_frac * T)
    t0 = t[-1] - w
    idx = t >= t0
    e = y[idx] - yref[idx]
    if e.size == 0:
        return float('nan'), float('nan')
    return float(np.mean(e)), float(np.sqrt(np.mean(e**2)))

# ====== Odporność na zakłócenia (SS okno) ======
def disturbance_robustness(t, e, Fw, window_frac=0.5, window_min=1.0):
    """
    Proste SNR: RMS(e)/RMS(Fw) w stanie ustalonym (ostatnie okno).
    Zwraca: snr (im mniejsze, tym lepiej), rms_e, rms_Fw, var_y.
    """
    t = np.asarray(t); e = np.asarray(e); Fw = np.asarray(Fw)
    T = t[-1] - t[0] if len(t) > 1 else 0.0
    w = max(window_min, window_frac * T)
    t0 = t[-1] - w
    idx = t >= t0
    if not np.any(idx):
        return float('nan'), float('nan'), float('nan')
    e_win = e[idx]; F_win = Fw[idx]
    rms_e = float(np.sqrt(np.mean(e_win**2)))
    rms_F = float(np.sqrt(np.mean(F_win**2))) if np.any(np.abs(F_win) > 0) else 0.0
    snr = float(rms_e / (rms_F + 1e-12)) if rms_F > 0 else float('nan')
    return snr, rms_e, rms_F

# ====== Zwięzłe podsumowanie na stdout ======
def print_summary(metrics: dict):
    # Minimalny, czytelny skrót
    parts = [
        f"MSE(th)={metrics.get('mse_theta', float('nan')):.6f}",
        f"MAE(th)={metrics.get('mae_theta', float('nan')):.6f}",
        f"MSE(x)={metrics.get('mse_x', float('nan')):.6f}",
        f"MAE(x)={metrics.get('mae_x', float('nan')):.6f}",
        f"IAE(th)={metrics.get('iae_theta', float('nan')):.4f}",
        f"ISE(th)={metrics.get('ise_theta', float('nan')):.4f}",
        f"IAE(x)={metrics.get('iae_x', float('nan')):.4f}",
        f"ISE(x)={metrics.get('ise_x', float('nan')):.4f}",
        f"E_u(L2)={metrics.get('e_u_l2', float('nan')):.4f}",
        f"E_u(L1)={metrics.get('e_u_l1', float('nan')):.4f}",
        f"t_s(th)={metrics.get('t_s_theta', float('nan')):.3f}s",
        f"t_s(x)={metrics.get('t_s_x', float('nan')):.3f}s",
        f"OS(th)={metrics.get('overshoot_theta', float('nan')):.2f}%",
        f"OS(x)={metrics.get('overshoot_x', float('nan')):.2f}%",
        f"ess(th)={metrics.get('ess_theta', float('nan')):.5f}",
        f"ess(x)={metrics.get('ess_x', float('nan')):.5f}",
        f"SNR_th={metrics.get('snr_theta', float('nan')):.3g}",
        f"SNR_x={metrics.get('snr_x', float('nan')):.3g}",
        f"T_sim={metrics.get('sim_time_wall', float('nan')):.3f}s",
        f"T_ctrl={metrics.get('ctrl_time_total', float('nan')):.3f}s",
    ]
    print("  ".join(parts))
