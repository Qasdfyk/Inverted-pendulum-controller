
import numpy as np
from env.model import f_nonlinear, linearize_upright

def test_rhs_finite():
    pars = {"M":2.4,"m":0.23,"l":0.36,"g":9.81}
    x = np.array([0.01,0.0,0.0,0.0])
    dx = f_nonlinear(x, u=0.0, pars=pars, Fw=0.0)
    assert np.all(np.isfinite(dx))

def test_linearization_jacobian_close():
    pars = {"M":2.4,"m":0.23,"l":0.36,"g":9.81}
    A,B = linearize_upright(**pars)
    x0 = np.zeros(4); u0 = 0.0
    eps = 1e-6
    An = np.zeros_like(A); Bn = np.zeros_like(B)
    for i in range(4):
        dx = np.zeros(4); dx[i]=eps
        f1 = f_nonlinear(x0+dx, u0, pars, 0.0)
        f0 = f_nonlinear(x0-dx, u0, pars, 0.0)
        An[:,i] = (f1 - f0)/(2*eps)
    du = 1e-6
    f1 = f_nonlinear(x0, u0+du, pars, 0.0)
    f0 = f_nonlinear(x0, u0-du, pars, 0.0)
    Bn[:,0] = (f1 - f0)/(2*du)
    assert np.allclose(An, A, atol=1e-1)
    assert np.allclose(Bn, B, atol=1e-1)
