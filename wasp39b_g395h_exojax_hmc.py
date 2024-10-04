import G395H_resolution
import TP_funcs
import cloud_funcs
import make_opa
from wasp39b_params import period_day, Mp_mean, Mp_std, Rstar_mean, Rstar_std

import os
import sys
import numpy as np

from jax import random
import jax.numpy as jnp

from exojax.spec.atmrt import ArtTransPure
from exojax.utils.constants import RJ, Rs, MJ

from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.utils.astrofunc import gravity_jupiter

from exojax.spec.unitconvert import wav2nu
from exojax.utils.grids import wavenumber_grid

from exojax.spec.specop import SopRotation
from exojax.spec.specop import SopInstProfile

# PPL
from numpyro.infer import Predictive
from numpyro.infer import MCMC, NUTS
import numpyro
import numpyro.distributions as dist

# from jax import config
# config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)


def make_sops(nu_grid):
    sop_rot = SopRotation(nu_grid, vsini_max=100.0)
    sop_inst = SopInstProfile(nu_grid, vrmax=300.0)
    return [sop_rot, sop_inst]


def make_grid(split, inst_nus_list):
    rinst_multiplier = 10
    wav_margin = 10
    nu_grid_list = []
    beta_inst_list = []
    sop_list = []
    for i in range(split):
        wav_split = wav2nu(inst_nus_list[i], unit="nm")
        Rinst = G395H_resolution.res_G395H(np.mean(wav_split))

        Nx = np.log(
            (np.max(wav_split) + wav_margin) / (np.min(wav_split) - wav_margin)
        ) / np.log((Rinst * rinst_multiplier + 0.5) / (Rinst * rinst_multiplier - 0.5))
        Nx = int(Nx // 2 * 2)
        nu_grid, wav, res = wavenumber_grid(
            np.min(wav_split) - wav_margin,
            np.max(wav_split) + wav_margin,
            N=Nx,
            unit="nm",
            xsmode="premodit",
        )
        print("Resolution", res)
        # print("Wavelength (nm)", wav / 10)

        nu_grid_list.append(nu_grid)
        beta_inst_list.append(resolution_to_gaussian_std(Rinst))
        # Spectral operator on rotation
        # Spectral operator on Instrumental profile and sampling
        sop_list.append(make_sops(nu_grid_list[i]))
        print("Rinst", Rinst)
        print("beta_inst_list", beta_inst_list)
    return nu_grid_list, beta_inst_list, sop_list


def dtau_cia(art, mmw_arr, vmr1_arr, vmr2_arr, opa_cia, Tarr, gravity):
    logacia_matrix = opa_cia.logacia_matrix(Tarr)
    dtaucia = art.opacity_profile_cia(
        logacia_matrix, Tarr, vmr1_arr, vmr2_arr, mmw_arr[:, None], gravity
    )
    return dtaucia


def dtau_mol(art, mmw_arr, vmr_arr, opa, Tarr, gravity):
    # vmr_arr = art.constant_mmr_profile(vmr)
    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    dtau = art.opacity_profile_xs(xsmatrix, vmr_arr, mmw_arr[:, None], gravity)
    return dtau


def rp_from_dtau(
    vmr_arr,
    Tarr,
    dtau_cloud,
    art,
    opa_cias,
    opa_mols,
    mmw,
    vmrH2,
    vmrHe,
    radius_btm,
    gravity_btm,
    gravity,
):
    dtau = dtau_cloud
    dtau += dtau_cia(art, mmw, vmrH2, vmrH2, opa_cias[0], Tarr, gravity_btm)
    dtau += dtau_cia(art, mmw, vmrH2, vmrHe, opa_cias[1], Tarr, gravity_btm)
    for i in range(vmr_arr.shape[0]):
        dtau += dtau_mol(art, mmw, vmr_arr[i], opa_mols[i], Tarr, gravity)
    rp2 = art.run(dtau, Tarr, mmw, radius_btm, gravity_btm)
    return rp2


def frun(
    Mp,
    art,
    radius_btm,
    vmr_arr,
    vmrH2,
    vmrHe,
    mmw,
    Tarr,
    dtau_cloud,
    opa_mols,
    opa_cias,
):
    gravity_btm = gravity_jupiter(radius_btm / RJ, Mp / MJ)
    gravity = art.gravity_profile(Tarr, mmw, radius_btm, gravity_btm)
    Rp2 = rp_from_dtau(
        vmr_arr,
        Tarr,
        dtau_cloud,
        art,
        opa_cias,
        opa_mols,
        mmw,
        vmrH2,
        vmrHe,
        radius_btm,
        gravity_btm,
        gravity,
    )
    return Rp2


def sampling(
    Rp2,
    radius_star,
    RV,
    radius_btm,
    sops,
    beta_inst,
    inst_nus,
):
    sop_rot, sop_inst = sops
    vsini = 2 * jnp.pi * radius_btm / (period_day * 24 * 60 * 60) / 100000  # cm -> km
    u1 = 0
    u2 = 0
    Frot = sop_rot.rigid_rotation(Rp2, vsini, u1, u2)
    Frot_inst = sop_inst.ipgauss(Frot, beta_inst)
    # Frot_inst = sop_inst.ipgauss(Rp2, beta_inst)
    Rp2_sample = sop_inst.sampling(Frot_inst, RV, inst_nus)
    return jnp.sqrt(Rp2_sample) * (radius_btm / radius_star)


if __name__ == "__main__":
    dir_save = "output/"
    os.makedirs(dir_save, exist_ok=True)

    wavelength_raw = np.load("data/wavelength.npy")
    rp_mean = np.load("data/wasp39b_nirspec_g395h_rp_mean.npy")
    rp_std = np.load("data/wasp39b_nirspec_g395h_rp_std.npy")

    num_warmup, num_samples = 1000, 1500

    TP_model = "constant"  # constant, powerlaw, spline, guillot2010
    cloud_model = "gray"  # gray, spline, None

    diffmode = 0

    TP_dict = {
        "constant": TP_funcs.TP_constant_sample,
    }
    TP_func = TP_dict[TP_model]

    cloud_sample_dict = {
        "no": cloud_funcs.nocloud_sample,
        "gray": cloud_funcs.cloud_gray_sample,
    }
    cloud_dtau_dict = {
        "no": cloud_funcs.nocloud_dtau,
        "gray": cloud_funcs.cloud_gray_dtau,
    }
    cloud_params_func = cloud_sample_dict[cloud_model]
    cloud_dtau_func = cloud_dtau_dict[cloud_model]

    # Wavenumber Grid
    # For sampling with different wavelengths
    split = 5
    inst_nus_list = np.array_split(wav2nu(wavelength_raw, "nm"), split)
    nu_grid_list, beta_inst_list, sop_list = make_grid(split, inst_nus_list)

    art = ArtTransPure(pressure_top=1.0e-11, pressure_btm=1.0e1, nlayer=120)
    Tlow = 500.0
    Thigh = 2000.0
    art.change_temperature_range(Tlow, Thigh)

    ciamols, opa_cias_list = make_opa.opa_cias_list(nu_grid_list)
    mols, opa_mols_list, molmass_arr = make_opa.opa_mols_list(
        nu_grid_list, Tlow, Thigh, diffmode=diffmode, save_pickle=False
    )
    # mols, opa_mols_list, molmass_arr = make_opa.opa_mols_list_pickle()

    def model_c(rp_mean, rp_std):
        Mp_tmp = numpyro.sample(
            "Mp_tmp", dist.TruncatedNormal(0, 1, low=-Mp_mean / Mp_std)
        )
        Mp = numpyro.deterministic("Mp", Mp_tmp * Mp_std + Mp_mean) * MJ
        radius_star_tmp = numpyro.sample(
            "Rs_tmp", dist.TruncatedNormal(0, 1, low=-Rstar_mean / Rstar_std)
        )
        radius_star = (
            numpyro.deterministic("Rs", radius_star_tmp * Rstar_std + Rstar_mean) * Rs
        )
        RV = numpyro.sample("RV", dist.Uniform(-200, 0))
        radius_btm = numpyro.sample("Radius_btm", dist.Uniform(1.0, 1.50)) * RJ

        # VMR profile
        vmr_arr = []
        for mol in mols:
            logVMR = numpyro.sample("logVMR_" + mol, dist.Uniform(-15, 0))
            vmr_arr.append(art.constant_mmr_profile(jnp.power(10, logVMR)))
        vmr_arr = jnp.array(vmr_arr)
        mmw = make_opa.calc_mmw(vmr_arr, molmass_arr)
        vmrH2, vmrHe = make_opa.vmrH2_vmrHe(vmr_arr)

        Tarr = TP_func(art, Tlow, Thigh)
        params_cloud = cloud_params_func()

        vals = [radius_btm, vmr_arr, vmrH2, vmrHe, mmw, Tarr]
        mu = jnp.array([])
        for i in range(split):
            dtau_cloud = cloud_dtau_func(params_cloud, art, nu_grid_list[i])
            rp2 = frun(
                Mp,
                art,
                *vals,
                dtau_cloud,
                opa_mols=opa_mols_list[i],
                opa_cias=opa_cias_list[i],
            )
            mu_tmp = sampling(
                rp2,
                radius_star,
                RV,
                radius_btm,
                sops=sop_list[i],
                beta_inst=beta_inst_list[i],
                inst_nus=inst_nus_list[i],
            )
            mu = jnp.concatenate([mu, mu_tmp])

        numpyro.sample(
            "rp",
            dist.Normal(mu[::-1], rp_std),
            obs=rp_mean,
        )

    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    kernel = NUTS(
        model_c,
        forward_mode_differentiation=False,
        max_tree_depth=13,
    )

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key_, rp_mean=rp_mean, rp_std=rp_std)
    mcmc.print_summary()
    with open(dir_save + "mcmc_summary.txt", "w") as f:
        # save current stdout
        original_stdout = sys.stdout
        # redirect stdout to a file
        sys.stdout = f
        # write output to file
        mcmc.print_summary()
        # restore the stdout
        sys.stdout = original_stdout

    posterior_sample = mcmc.get_samples()
    jnp.savez(dir_save + "posterior_sample", **posterior_sample)

    pred = Predictive(model_c, posterior_sample, return_sites=["rp"])
    predictions = pred(rng_key_, rp_mean=None, rp_std=rp_std)
    jnp.save(dir_save + "rp_pred", predictions["rp"])
