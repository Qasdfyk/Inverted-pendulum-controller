import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_cartpole(t, X, params=None, speed=1.0):
    """
    Clean cart–pole animation with wheels, smooth camera, and robust drawing.
    Avoids 'blit' to prevent backend issues; keeps animation object alive.
    """
    th = X[:, 0]
    x  = X[:, 2]

    p = params or {}
    l = p.get("l", 0.36)
    cart_w, cart_h = 0.35, 0.18
    wheel_r = 0.05
    pole_len = l * 1.5
    pad = 0.8

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.grid(True, alpha=0.3)
    ax.set_title("Cart–Pole")

    xmin = float(np.min(x) - pad)
    xmax = float(np.max(x) + pad)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-(wheel_r + 0.25), pole_len + 0.45)

    # Ground line
    ax.plot([xmin, xmax], [0, 0], color='k', lw=1, alpha=0.6)

    # Cart body & wheels
    cart = plt.Rectangle((x[0] - cart_w/2, wheel_r), cart_w, cart_h, ec='k', fc='#e6f3ff', lw=1.5)
    ax.add_patch(cart)
    wheel1 = plt.Circle((x[0] - cart_w/3, wheel_r), wheel_r, ec='k', fc='#333333')
    wheel2 = plt.Circle((x[0] + cart_w/3, wheel_r), wheel_r, ec='k', fc='#333333')
    ax.add_patch(wheel1); ax.add_patch(wheel2)

    # Pole & trail
    pole_line, = ax.plot([], [], lw=3, solid_capstyle='round', alpha=0.9)
    trail_line, = ax.plot([], [], lw=1, alpha=0.4)

    def pole_end(i):
        cx = x[i]
        cy = wheel_r + cart_h
        px = cx + pole_len * np.sin(th[i])
        py = cy + pole_len * np.cos(th[i])
        return cx, cy, px, py

    def init():
        cx, cy, px, py = pole_end(0)
        cart.set_x(x[0] - cart_w/2)
        wheel1.center = (x[0] - cart_w/3, wheel_r)
        wheel2.center = (x[0] + cart_w/3, wheel_r)
        pole_line.set_data([cx, px], [cy, py])
        trail_line.set_data([], [])
        return cart, wheel1, wheel2, pole_line, trail_line

    def update(i):
        cx, cy, px, py = pole_end(i)
        cart.set_x(cx - cart_w/2)
        wheel1.center = (cx - cart_w/3, wheel_r)
        wheel2.center = (cx + cart_w/3, wheel_r)
        pole_line.set_data([cx, px], [cy, py])
        j0 = max(0, i - 150)
        trail_line.set_data(x[j0:i+1], (wheel_r + 0.01) * np.ones(i + 1 - j0))
        ax.set_xlim(cx - pad, cx + pad)  # camera follow
        return cart, wheel1, wheel2, pole_line, trail_line

    interval_ms = max(10, int(1000 * (t[1] - t[0]) / max(speed, 1e-6)))
    ani = FuncAnimation(fig, update, frames=len(t), init_func=init,
                        interval=interval_ms, blit=False)
    # Keep a reference so it doesn't get GC'd before show()
    setattr(fig, "_cartpole_ani", ani)
    plt.show()
