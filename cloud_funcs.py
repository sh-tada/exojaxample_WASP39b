import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def nocloud_sample(limb=""):
    """No cloud (dtau=0)"""
    return 0.0


def nocloud_dtau(zero, art, nu_grid):
    """No cloud (dtau=0)"""
    return 0.0


def cloud_gray_sample(limb=""):
    """gray cloud (dtau=1 under lodP_cloud)"""
    logP_cloud = numpyro.sample("logP_cloud", dist.Uniform(-11, 1))
    return logP_cloud


def cloud_gray_dtau(logP_cloud, art, nu_grid):
    """gray cloud (dtau=1 under lodP_cloud)"""
    if logP_cloud is None:
        return 0.0
    else:
        # the width of sigmoid
        width = 1.0 / 25.0
        # dtau of cloud
        dtau = 50.0

        # add offset to make tau(P_cloud) = 1.0
        logP_cloud_arr = (logP_cloud + width * jnp.log(dtau - 1.0)) * jnp.ones_like(
            nu_grid
        )
        pressure_arr = jnp.log10(art.pressure)

        # sigmoid
        # clip to avoid overflow
        dtau_cloud = dtau / (
            1.0
            + jnp.exp(
                -jnp.clip(
                    (pressure_arr[:, None] - logP_cloud_arr[None, :]) / width, -50, 50
                )
            )
        )
        return dtau_cloud


if __name__ == "__main__":
    pass
