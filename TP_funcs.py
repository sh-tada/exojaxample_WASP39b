import numpy as np

# from spline import spline
import numpyro
import numpyro.distributions as dist


def TP_constant(art, T0):
    """Constant temperature profile"""
    return T0 * np.ones_like(art.pressure)


def TP_constant_sample(art, Tlow, Thigh):
    """Constant temperature profile"""
    T0 = numpyro.sample("T0", dist.Uniform(Tlow, Thigh))
    return TP_constant(art, T0)


if __name__ == "__main__":
    pass
