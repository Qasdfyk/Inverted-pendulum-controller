"""Figures and LaTeX tables for results_report/ from results/cost_study (see run_cost_study.py)."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from study_setup import all_names, BASELINE, NMPC_VARIANTS, PARAMS, S2_T_ON, S2_LEVELS, DT
from run_cost_study import OUT_DIR, theta_ss
from mpc_cost import RHO, FULL_NAMES

REPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results_report'))
FIG_DIR = os.path.join(REPORT_DIR, 'figs')
TAB_DIR = os.path.join(REPORT_DIR, 'tables')

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'lines.linewidth': 1.6,
})

SCEN_TITLE = {"S1": "Scenariusz S1 (nominalny)", "S2": "Scenariusz S2 (skoki zakłócenia)"}
STATUS_MARK = {"stable": "", "not settled": "$^{\\ddagger}$", "unstable": "$^{\\dagger}$"}


def load():
    res = json.load(open(os.path.join(OUT_DIR, "results.json"), encoding="utf-8"))
    traj = {}
    for key in res:
        s, n = key.split("|")
        d = np.load(os.path.join(OUT_DIR, "traj", f"{s}_{n}.npz"))
        traj[key] = (d["X"], d["U"], d["Fw"])
    return res, traj


def refs(scen, X, Fw):
    k = np.arange(len(X))
    if scen == "S1":
        return k, np.zeros(len(X)), np.full(len(X), 0.1)
    # theta_ss follows the disturbance level acting at each instant
    Fw_full = np.append(Fw, Fw[-1])
    return k, theta_ss(Fw_full), np.zeros(len(X))


def fname(name):
    return name.replace("-", "_").lower()


def plot_single(scen, name, X, U, Fw):
    k, th_ref, x_ref = refs(scen, X, Fw)
    rows = 4 if scen == "S2" else 3
    fig = plt.figure(figsize=(10, 2.0 * rows + 2.4))
    gs = fig.add_gridspec(rows + 1, 3, height_ratios=[1] * rows + [1.1])
    ax_th = fig.add_subplot(gs[0, :]); ax_x = fig.add_subplot(gs[1, :], sharex=ax_th)
    ax_u = fig.add_subplot(gs[2, :], sharex=ax_th)
    ax_th.plot(k, X[:, 0], label=name)
    ax_th.plot(k, th_ref, 'k--', lw=1, label=r'$\theta_\mathrm{ss}$' if scen == "S2" else 'Wartość zadana')
    ax_th.set_ylabel(r'$\theta$ [rad]'); ax_th.legend(loc='upper right')
    ax_x.plot(k, X[:, 2]); ax_x.plot(k, x_ref, 'k--', lw=1); ax_x.set_ylabel(r'$x_\mathrm{p}$ [m]')
    ax_u.step(k[:-1], U, where='post'); ax_u.set_ylabel(r'$u$ [N]')
    if scen == "S2":
        ax_f = fig.add_subplot(gs[3, :], sharex=ax_th)
        ax_f.step(k[:-1], Fw, where='post', color='tab:red'); ax_f.set_ylabel(r'$F_\mathrm{w}$ [N]')
        last = ax_f
    else:
        last = ax_u
    last.set_xlabel('$k$')
    for a in fig.axes:
        a.grid(True, alpha=0.4)
    # Histograms: control errors and control signal
    e_th = th_ref[1:] - X[1:, 0]; e_x = x_ref[1:] - X[1:, 2]
    for j, (data, lab) in enumerate([(e_th, r'$\theta_\mathrm{ref}-\theta$ [rad]'),
                                     (e_x, r'$x_\mathrm{ref}-x_\mathrm{p}$ [m]'), (U, r'$u$ [N]')]):
        ah = fig.add_subplot(gs[rows, j]); ah.hist(data, bins=40, color='tab:gray')
        ah.set_xlabel(lab); ah.set_ylabel('liczność' if j == 0 else ''); ah.grid(True, alpha=0.4)
    if scen == "S2":
        ax_th.set_title(r'$\theta_\mathrm{ss}=-\arctan(F_\mathrm{w}/(mg))$ (histogram błędu kąta liczony względem $\theta_\mathrm{ss}$)',
                        fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"{scen}_{fname(name)}.pdf")
    fig.savefig(path); plt.close(fig)


def plot_compare(scen, names, traj, tag, title):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for n in names:
        X, U, Fw = traj[f"{scen}|{n}"]
        k = np.arange(len(X))
        axs[0].plot(k, X[:, 0], label=n); axs[1].plot(k, X[:, 2]); axs[2].step(k[:-1], U, where='post')
    X, U, Fw = traj[f"{scen}|{names[0]}"]
    k, th_ref, x_ref = refs(scen, X, Fw)
    axs[0].plot(k, th_ref, 'k--', lw=1, label=r'$\theta_\mathrm{ss}$' if scen == "S2" else 'Wartość zadana')
    axs[1].plot(k, x_ref, 'k--', lw=1)
    axs[0].set_ylabel(r'$\theta$ [rad]'); axs[1].set_ylabel(r'$x_\mathrm{p}$ [m]'); axs[2].set_ylabel(r'$u$ [N]')
    axs[2].set_xlabel('$k$'); axs[0].legend(ncol=4, loc='upper right'); axs[0].set_title(title)
    for a in axs:
        a.grid(True, alpha=0.4)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, f"{scen}_compare_{tag}.pdf")); plt.close(fig)


def plot_rho():
    x = np.linspace(-2, 2, 801)
    groups = [["L2", "L1", "Ch", "F"], ["Hu", "MH", "K", "L"], ["M", "Z", "BH", "Be"], ["C", "MS", "R", "Y"]]
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.5))
    for ax, g in zip(axs.ravel(), groups):
        for key in g:
            ax.plot(x, RHO[key](x, 1.0), label=FULL_NAMES[key].replace('\\"u', 'ü').replace('$_2$', '$_2$'))
        ax.set_ylim(0, 4); ax.set_xlabel(r'$\Delta u$'); ax.set_ylabel(r'$\rho(\Delta u)$')
        ax.legend(); ax.grid(True, alpha=0.4)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "rho_functions.pdf")); plt.close(fig)


def sci(v):
    if not np.isfinite(v):
        return "--"
    if v == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(v))))
    return f"${v / 10**e:.4f}\\times10^{{{e}}}$"


def label(n):
    return n.replace("NMPC-L2", "NMPC-L$_2$").replace("NMPC-L1", "NMPC-L$_1$")


def table_indicators(scen, res):
    cols = [("mse_theta", r"$\mathrm{MSE}_\theta$"), ("mse_x", r"$\mathrm{MSE}_{x}$"),
            ("mae_theta", r"$\mathrm{MAE}_\theta$"), ("mae_x", r"$\mathrm{MAE}_{x}$"), ("kvt", r"$K_\mathrm{VT}$")]
    stable = [r for k, r in res.items() if k.startswith(scen + "|") and r["status"] == "stable"]
    best = {c: min(r[c] for r in stable) for c, _ in cols} if stable else {}
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\hline",
             "Algorytm & " + " & ".join(h for _, h in cols) + r" \\", r"\hline"]
    for group in (BASELINE, [f"NMPC-{k}" for k in NMPC_VARIANTS]):
        for n in group:
            r = res.get(f"{scen}|{n}")
            if r is None:
                continue
            cells = []
            for c, _ in cols:
                s = sci(r[c])
                if r["status"] == "stable" and np.isclose(r[c], best.get(c, np.nan)):
                    s = r"{\boldmath" + s + "}"
                cells.append(s)
            lines.append(label(n) + STATUS_MARK[r["status"]] + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    open(os.path.join(TAB_DIR, f"{scen}_indicators.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def table_extra(res):
    lines = [r"\begin{tabular}{lccccc}", r"\hline",
             r"Algorytm & status S1 & status S2 & $\max|u|$ S2 [N] & $\max|x_\mathrm{p}|$ S2 [m] & "
             r"niepowodzenia solvera \\", r"\hline"]
    status_pl = {"stable": "stabilny", "not settled": "nieustalony", "unstable": "niestabilny"}
    for n in all_names():
        r1, r2 = res.get(f"S1|{n}"), res.get(f"S2|{n}")
        if r1 is None or r2 is None:
            continue
        fails = f"{r1['solver_fails'] + r2['solver_fails']}/{r1['solver_calls'] + r2['solver_calls']}" \
            if r1["solver_calls"] else "--"
        lines.append(f"{label(n)} & {status_pl[r1['status']]} & {status_pl[r2['status']]} & "
                     f"{r2['max_u']:.2f} & {r2['max_x']:.3f} & {fails} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    open(os.path.join(TAB_DIR, "status.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def table_params():
    p = PARAMS
    fmt = lambda v: ", ".join(f"{x:g}" for x in v)
    rows = [
        ("PID", rf"$K_{{p,\theta}}={p['PID']['ang_pid']['Kp']:g}$, $K_{{i,\theta}}={p['PID']['ang_pid']['Ki']:g}$, "
                rf"$K_{{d,\theta}}={p['PID']['ang_pid']['Kd']:g}$; $K_{{p,x}}={p['PID']['cart_pid']['Kp']:g}$, "
                rf"$K_{{i,x}}={p['PID']['cart_pid']['Ki']:g}$, $K_{{d,x}}={p['PID']['cart_pid']['Kd']:g}$; $I_{{\max}}=5$"),
        ("LQR", rf"$Q=\mathrm{{diag}}({fmt(p['LQR']['Q'])})$, $R={p['LQR']['R']:g}$"),
        ("LQR-I", rf"$\tilde{{Q}}=\mathrm{{diag}}({fmt(p['LQR-I']['Q'])})$, $R={p['LQR-I']['R']:g}$"),
        ("PID-LQR", rf"$Q=\mathrm{{diag}}({fmt(p['PID-LQR']['lqr_gains']['Q'])})$, $R={p['PID-LQR']['lqr_gains']['R']:g}$; "
                    rf"$K_{{p,x}}={p['PID-LQR']['pid_gains']['Kp']:g}$, $K_{{i,x}}={p['PID-LQR']['pid_gains']['Ki']:g}$, "
                    rf"$K_{{d,x}}={p['PID-LQR']['pid_gains']['Kd']:g}$"),
        ("Fuzzy-LQR", r"$K_\mathrm{base}=[40,1,15,20]$, przyrosty $[20,5,5,2]\cdot L_i$, $G=0.36$, "
                      r"$K_\mathrm{LQR}$: $Q=I$, $R=1$"),
        ("LMPC", rf"$N={p['LMPC']['N']}$, $N_\mathrm{{u}}={p['LMPC']['Nu']}$, $Q=\mathrm{{diag}}({fmt(p['LMPC']['Q'])})$, $R={p['LMPC']['R']:g}$"),
        ("MPC-NO-alt", rf"$N=12$, $N_\mathrm{{u}}=4$, $Q$ jak NMPC, $R={p['MPC-NO-alt']['R']:g}$, $R_\mathrm{{abs}}={p['MPC-NO-alt']['R_abs']:g}$"),
        ("NMPC-$\\rho$", rf"$N=12$, $N_\mathrm{{u}}=4$, $Q=\mathrm{{diag}}({fmt(np.round(p['NMPC']['Q'], 2))})$, "
                         rf"$R={p['NMPC']['R']:g}$, $c={p['NMPC']['c']:g}$ (wszystkie $\rho$)"),
    ]
    lines = [r"\begin{tabular}{lp{12.5cm}}", r"\hline", r"Algorytm & Parametry \\", r"\hline"]
    lines += [f"{a} & {b} \\\\" for a, b in rows]
    lines += [r"\hline", r"\end{tabular}"]
    open(os.path.join(TAB_DIR, "params.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def figures_tex(res):
    """One float per controller and scenario, in table order."""
    out = []
    for scen in ("S1", "S2"):
        out.append(f"\\subsection{{Przebiegi -- {SCEN_TITLE[scen]}}}")
        for n in all_names():
            r = res.get(f"{scen}|{n}")
            if r is None:
                continue
            status_pl = {"stable": "stabilny", "not settled": "\\textbf{nieustalony}", "unstable": "\\textbf{NIESTABILNY}"}
            out.append("\\begin{figure}[p]\\centering")
            out.append(f"\\includegraphics[width=\\linewidth]{{figs/{scen}_{fname(n)}.pdf}}")
            out.append(f"\\caption{{{label(n)}, {SCEN_TITLE[scen]}: status automatyczny -- {status_pl[r['status']]}.}}")
            out.append("\\end{figure}\n\\clearpage")
    open(os.path.join(TAB_DIR, "all_figures.tex"), "w", encoding="utf-8").write("\n".join(out) + "\n")


def main():
    os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(TAB_DIR, exist_ok=True)
    res, traj = load()
    plot_rho()
    for key, (X, U, Fw) in traj.items():
        s, n = key.split("|")
        plot_single(s, n, X, U, Fw)
    for scen in ("S1", "S2"):
        table_indicators(scen, res)
        plot_compare(scen, BASELINE + ["NMPC-L2"], traj, "baseline", "Regulatory bazowe i NMPC-L$_2$")
        stable = [n for n in (f"NMPC-{k}" for k in NMPC_VARIANTS[1:])
                  if res.get(f"{scen}|{n}", {}).get("status") == "stable"]
        top = sorted(stable, key=lambda n: res[f"{scen}|{n}"]["mse_theta"])[:3]
        low_kvt = sorted(stable, key=lambda n: res[f"{scen}|{n}"]["kvt"])[:1]
        pick = ["NMPC-L2"] + top + [n for n in low_kvt if n not in top]
        plot_compare(scen, pick, traj, "nmpc", "NMPC-L$_2$, trzy najlepsze wg MSE$_\\theta$ i najniższy $K_\\mathrm{VT}$")
        macro = {"S1": "PickedSA", "S2": "PickedSB"}[scen]
        open(os.path.join(TAB_DIR, f"{scen}_picked.tex"), "w", encoding="utf-8").write(
            "\\newcommand{\\" + macro + "}{" + ", ".join(label(n) for n in pick) + "}\n")
    levels = [f"$F_\\mathrm{{w}}={lv:g}$~N od $t={t:g}$~s" for t, lv in zip(S2_T_ON, S2_LEVELS)]
    open(os.path.join(TAB_DIR, "s2_levels.tex"), "w", encoding="utf-8").write(
        "$F_\\mathrm{w}=0$ dla $t<" + f"{S2_T_ON[0]:g}" + "$~s, następnie " + "; ".join(levels) + ".\n")
    table_extra(res)
    table_params()
    figures_tex(res)
    print("report assets written to", REPORT_DIR)


if __name__ == "__main__":
    main()
