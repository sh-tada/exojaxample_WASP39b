from wasp39b_g395h_exojax_hmc import make_grid, frun, sampling

import TP_funcs
import cloud_funcs
import make_opa

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

import jax.numpy as jnp


from exojax.spec.atmrt import ArtTransPure
from exojax.utils.constants import RJ, Rs, MJ
from exojax.utils.astrofunc import gravity_jupiter
from exojax.spec.unitconvert import wav2nu

# PPL
import arviz
import corner
from numpyro.diagnostics import hpdi

from scipy.interpolate import interp1d


# from jax import config
# config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)

if __name__ == "__main__":
    fontsize = 24
    plt.rcParams["font.size"] = fontsize

    dir_save = "output/"
    os.makedirs(dir_save, exist_ok=True)

    wavelength_raw = np.load("data/wavelength.npy")
    rp_median = np.load("data/wasp39b_nirspec_g395h_rp_median.npy")
    rp_hpdi = np.load("data/wasp39b_nirspec_g395h_rp_hpdi_68.npy")
    rp_yerr = np.array([rp_median - rp_hpdi[0], rp_hpdi[1] - rp_median])

    TP_model = "constant"  # constant, powerlaw, spline, guillot2010
    cloud_model = "gray"  # gray, spline, None

    diffmode = 0

    TP_dict = {
        "constant": TP_funcs.TP_constant,
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
    # mols, opa_mols_list, molmass_arr = make_opa.opa_mols_list(
    #     nu_grid_list, Tlow, Thigh, diffmode=diffmode, save_pickle=False
    # )
    with open("../OpaPremodit_dict.bin", "rb") as f:
        opa_mols_list = pickle.load(f)
    with open("../mols_list.bin", "rb") as f:
        mols = pickle.load(f)
    with open("../molmass_arr.bin", "rb") as f:
        molmass_arr = pickle.load(f)
    # mols, opa_mols_list, molmass_arr = make_opa.opa_mols_list_pickle()

    def model(params_dict, mols_plot=mols, cia=True, cloud=True):
        Mp = params_dict["Mp"] * MJ
        radius_star = params_dict["Rs"] * Rs
        RV = params_dict["RV"]
        radius_btm = params_dict["Radius_btm"] * RJ

        # VMR profile
        vmr_arr = []
        for mol in mols:
            logVMR = params_dict["logVMR_" + mol]
            vmr_arr.append(art.constant_mmr_profile(jnp.power(10, logVMR)))
        vmr_arr = jnp.array(vmr_arr)
        mmw = make_opa.calc_mmw(vmr_arr, molmass_arr)
        vmrH2, vmrHe = make_opa.vmrH2_vmrHe(vmr_arr)
        if not cia:
            vmrH2 = jnp.zeros_like(vmrH2)
            vmrHe = jnp.zeros_like(vmrHe)

        vmr_arr_masked = []
        opa_mols_list_masked = []
        for i, mol in enumerate(mols):
            if mol in mols_plot:
                vmr_arr_masked.append(vmr_arr[i])
        for j in range(len(opa_mols_list)):
            opa_mols_tmp = []
            for i, mol in enumerate(mols):
                if mol in mols_plot:
                    opa_mols_tmp.append(opa_mols_list[j][i])
            opa_mols_list_masked.append(opa_mols_tmp)
        vmr_arr_masked = jnp.array(vmr_arr_masked)

        Tarr = TP_func(art, params_dict["T0"])
        params_cloud = params_dict["logP_cloud"]

        if not cloud:
            params_cloud = None

        vals = [radius_btm, vmr_arr_masked, vmrH2, vmrHe, mmw, Tarr]
        mu = jnp.array([])
        for i in range(split):
            dtau_cloud = cloud_dtau_func(params_cloud, art, nu_grid_list[i])
            rp2 = frun(
                Mp,
                art,
                *vals,
                dtau_cloud,
                opa_mols=opa_mols_list_masked[i],
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

        return mu[::-1]

    def only(params_dict):
        only_models = {}
        for mol in mols:
            # select molecule
            only_models[mol] = model(
                params_dict, mols_plot=[mol], cia=False, cloud=False
            )
        only_models["CIA"] = model(params_dict, mols_plot=[], cia=True, cloud=False)
        if cloud_model != "no":
            only_models["cloud"] = model(
                params_dict, mols_plot=[], cia=False, cloud=True
            )
        return only_models

    def pressure_radius(params_dict):
        Mp = params_dict["Mp"] * MJ
        radius_btm = params_dict["Radius_btm"] * RJ

        # VMR profile
        vmr_arr = []
        for mol in mols:
            logVMR = params_dict["logVMR_" + mol]
            vmr_arr.append(art.constant_mmr_profile(jnp.power(10, logVMR)))
        vmr_arr = jnp.array(vmr_arr)
        mmw = make_opa.calc_mmw(vmr_arr, molmass_arr)

        Tarr = TP_func(art, params_dict["T0"])

        gravity_btm = gravity_jupiter(radius_btm / RJ, Mp / MJ)
        normalized_height, normalized_radius_lower = art.atmosphere_height(
            Tarr, mmw, radius_btm, gravity_btm
        )
        normalized_radius_mid = normalized_radius_lower + normalized_height / 2.0
        return jnp.log10(art.pressure), normalized_radius_mid * radius_btm

    def pressure_to_radius(logP, params_dict):
        radius_star = params_dict["Rs"] * Rs
        _, radius_mid = pressure_radius(params_dict)
        f = interp1d(np.log10(art.pressure), radius_mid, kind="linear")
        if logP >= np.min(np.log10(art.pressure)) and logP <= np.max(
            np.log10(art.pressure)
        ):
            radius = f(logP) / radius_star
            return radius
        else:
            return None

    posterior_sample = np.load(dir_save + "posterior_sample.npz")
    posterior_sample = {key: posterior_sample[key] for key in posterior_sample.files}
    posterior_sample_median = {
        key: np.median(posterior_sample[key]) for key in posterior_sample
    }

    prediction_rp = np.load(dir_save + "rp_pred.npy")
    prediction_rp_median = np.median(prediction_rp, axis=0)
    prediction_rp_hpdi = hpdi(prediction_rp, 0.95)

    offset_only = 0.01
    ymin = 0.138 - offset_only
    ymax = 0.165

    fig = plt.figure(figsize=(24, 9))
    ax = fig.add_subplot()
    ax.errorbar(
        wavelength_raw,
        rp_median,
        yerr=rp_yerr,
        fmt=".",
        color="C0",
        label=r"data",
        alpha=0.5,
        markersize=1.5,
        linewidth=0.3,
    )
    ax.fill_between(
        wavelength_raw,
        prediction_rp_hpdi[0],
        prediction_rp_hpdi[1],
        alpha=0.2,
        interpolate=True,
        color="gray",
        edgecolor="None",
        label="95% area",
    )

    median_model = model(posterior_sample_median)
    ax.plot(
        wavelength_raw,
        median_model,
        color="black",
        label="all",
        lw=1,
    )

    only_dic = only(posterior_sample_median)

    for label in only_dic:
        print(label, only_dic[label])
        print(
            posterior_sample_median["Radius_btm"] * RJ / (0.9 * Rs),
            np.max(only_dic[label]),
        )
        if (
            np.max(only_dic[label])
            <= posterior_sample_median["Radius_btm"] * RJ / (0.9 * Rs) + 0.0015
        ):
            pass
        else:
            label_mol = label
            alpha = 0.3
            if label == "(12C)(16O)":
                label_mol = "CO"
                color = "C1"
                alpha = 0.5
            if label == "(12C)(16O)2":
                label_mol = "CO$_2$"
                color = "C2"
            if label == "H2(16O)":
                label_mol = "H$_2$O"
                color = "navy"
                alpha = 0.3
            if label == "(12C)H4":
                label_mol = "CH$_4$"
                color = "C3"
            if label == "NH3":
                label_mol = "NH$_3$"
                # color = "C4"
                color = "gold"
            if label == "H2S":
                label_mol = "H$_2$S"
                color = "C6"
            if label == "SO2":
                label_mol = "SO$_2$"
                color = "C9"
                alpha = 0.5
            if label == "HCN":
                label_mol = "HCN"
                color = "C8"
            if label == "C2H2":
                label_mol = "C$_2$H$_2$"
                color = "C5"
            if label == "CIA":
                # color = "0.5"
                color = "None"
                ax.fill_between(
                    wavelength_raw,
                    ymin,
                    only_dic[label] - offset_only,
                    alpha=0.8,
                    color=color,
                    edgecolor="0.8",
                    # lw=0.5,
                    hatch="||",
                    label=label_mol,
                )
                # ax.plot(
                #     wavelength,
                #     only_dic[label],
                #     lw=0.5,
                #     # label=label_mol,
                #     color=color,
                #     zorder=2,
                # )
            elif label == "cloud":
                color = "None"
                ax.fill_between(
                    wavelength_raw,
                    ymin,
                    only_dic[label] - offset_only,
                    alpha=0.8,
                    color=color,
                    edgecolor="0.2",
                    hatch="x",
                    label=label_mol,
                )
            else:
                ax.fill_between(
                    wavelength_raw,
                    ymin,
                    only_dic[label] - offset_only,
                    alpha=alpha,
                    # lw=2,
                    label=label_mol,
                    color=color,
                    edgecolor="None",
                    # linestyle="--",
                )
            # ax.plot(
            #     wav_full_posterior,
            #     only_dic[label],
            #     lw=2,
            #     label=label_mol,
            #     color=cm((i + 1) / len(only_dic) * 0.9),
            #     zorder=2,
            # )

    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("$R_p$ [$R_s$]")
    ax.set_xlim(np.min(wavelength_raw), np.max(wavelength_raw))
    ax.set_ylim(ymin, ymax)
    # ax.set_title(r"WASP-39 b transmission spectrum from NIRSpec/G395H")
    ax.legend(ncol=5, fontsize=20)

    # 右側のy軸を追加
    ax2 = ax.twinx()
    ax2.set_ylabel("log$_{10}$(P [bar])")
    ax2.tick_params(axis="y")

    # 変換されたデータの目盛りをyの値に対応させて設定
    ax2.set_ylim(ax.get_ylim())  # ax1 (左の軸) と同じy範囲を使用
    # labels_tmp = [-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1]
    labels_tmp = [-10, -8, -6, -4, -2, 0]
    labels = []
    locs = []
    for i in labels_tmp:
        loc = pressure_to_radius(i, posterior_sample_median)
        print(i, loc)
        if loc is not None and loc >= ymin:
            labels += [i]
            locs += [loc]
    ax2.set_yticks(locs, labels)  # 変換されたラベルを右側の軸に設定

    plt.tight_layout()
    plt.savefig(dir_save + "rp_hmc.png")
    # plt.show()
    plt.close()

    plt.rcParams["font.size"] = 14.0
    posterior_sample[r"$R_\mathrm{s}$ [$\mathrm{R_{\odot}}$]"] = posterior_sample.pop(
        "Rs"
    )
    posterior_sample["$M_\mathrm{p}$ [$\mathrm{M_{J}}$]"] = posterior_sample.pop("Mp")
    posterior_sample["RV [$\mathrm{km/s}$]"] = posterior_sample.pop("RV")
    posterior_sample["$R_\mathrm{p}$ at 10 bar [$\mathrm{R_{J}}$]"] = (
        posterior_sample.pop("Radius_btm")
    )
    posterior_sample["T [K]"] = posterior_sample.pop("T0")
    posterior_sample["log$P_{\mathrm{cloud}}$"] = posterior_sample.pop("logP_cloud")
    posterior_sample["log$\mathrm{CO}$"] = posterior_sample.pop("logVMR_(12C)(16O)")
    posterior_sample["log$\mathrm{CO_2}$"] = posterior_sample.pop("logVMR_(12C)(16O)2")
    posterior_sample["log$\mathrm{CH_4}$"] = posterior_sample.pop("logVMR_(12C)H4")
    posterior_sample["log$\mathrm{C_2H_2}$"] = posterior_sample.pop("logVMR_C2H2")
    posterior_sample["log$\mathrm{H_2O}$"] = posterior_sample.pop("logVMR_H2(16O)")
    posterior_sample["log$\mathrm{H_2S}$"] = posterior_sample.pop("logVMR_H2S")
    posterior_sample["log$\mathrm{HCN}$"] = posterior_sample.pop("logVMR_HCN")
    posterior_sample["log$\mathrm{NH_3}$"] = posterior_sample.pop("logVMR_NH3")
    posterior_sample["log$\mathrm{SO_2}$"] = posterior_sample.pop("logVMR_SO2")

    new_order = [
        r"$R_\mathrm{s}$ [$\mathrm{R_{\odot}}$]",
        "$M_\mathrm{p}$ [$\mathrm{M_{J}}$]",
        "RV [$\mathrm{km/s}$]",
        "T [K]",
        "$R_\mathrm{p}$ at 10 bar [$\mathrm{R_{J}}$]",
        "log$P_{\mathrm{cloud}}$",
        "log$\mathrm{H_2O}$",
        "log$\mathrm{CO}$",
        "log$\mathrm{CO_2}$",
        "log$\mathrm{SO_2}$",
        "log$\mathrm{H_2S}$",
        "log$\mathrm{CH_4}$",
        "log$\mathrm{NH_3}$",
        "log$\mathrm{HCN}$",
        "log$\mathrm{C_2H_2}$",
    ]
    titles = [
        "$R_\mathrm{s}$",
        "$M_\mathrm{p}$",
        "RV",
        "T",
        "$R_\mathrm{p}$(10 bar)",
        "log$P_{\mathrm{cloud}}$",
        "log$\mathrm{H_2O}$",
        "log$\mathrm{CO}$",
        "log$\mathrm{CO_2}$",
        "log$\mathrm{SO_2}$",
        "log$\mathrm{H_2S}$",
        "log$\mathrm{CH_4}$",
        "log$\mathrm{NH_3}$",
        "log$\mathrm{HCN}$",
        "log$\mathrm{C_2H_2}$",
    ]
    posterior_sample = {key: posterior_sample[key] for key in new_order}
    arviz.plot_trace(
        posterior_sample,
        combined=False,
        # var_names=(),
        backend_kwargs={"constrained_layout": True},
    )
    plt.savefig(dir_save + "trace.png")
    plt.close()

    fig = corner.corner(
        posterior_sample,
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
        color="C0",
        label_kwargs={"fontsize": 20},
        titles=titles,
        title_kwargs={"fontsize": 14},
        smooth=1.0,
    )
    plt.savefig(dir_save + "corner.png")
    plt.close()

    h2o = np.power(10, posterior_sample["log$\mathrm{H_2O}$"])
    co = np.power(10, posterior_sample["log$\mathrm{CO}$"])
    co2 = np.power(10, posterior_sample["log$\mathrm{CO_2}$"])
    so2 = np.power(10, posterior_sample["log$\mathrm{SO_2}$"])
    h2s = np.power(10, posterior_sample["log$\mathrm{H_2S}$"])
    ch4 = np.power(10, posterior_sample["log$\mathrm{CH_4}$"])
    nh3 = np.power(10, posterior_sample["log$\mathrm{NH_3}$"])
    hcn = np.power(10, posterior_sample["log$\mathrm{HCN}$"])
    c2h2 = np.power(10, posterior_sample["log$\mathrm{C_2H_2}$"])
    h2 = 1 - (h2o + co + co2 + so2 + h2s + ch4 + nh3 + hcn + c2h2)

    c_h = (co + co2 + ch4 + hcn + 2 * c2h2) / (
        2 * h2o + 2 * h2s + 4 * ch4 + 3 * nh3 + hcn + 2 * c2h2 + 2 * h2
    )
    o_h = (h2o + co + 2 * co2 + 2 * so2) / (
        2 * h2o + 2 * h2s + 4 * ch4 + 3 * nh3 + hcn + 2 * c2h2 + 2 * h2
    )
    s_h = (h2s + so2) / (
        2 * h2o + 2 * h2s + 4 * ch4 + 3 * nh3 + hcn + 2 * c2h2 + 2 * h2
    )
    c_o = c_h / o_h
    s_o = s_h / o_h
    m_h = (
        c_h
        + o_h
        + s_h
        + (3 * nh3 + hcn)
        / (2 * h2o + 2 * h2s + 4 * ch4 + 3 * nh3 + hcn + 2 * c2h2 + 2 * h2)
    )

    log_solar = [-3.54, -3.31, -4.88, -0.23, -1.57, 0.00]
    name = [
        "$\mathrm{C/H}$",
        "$\mathrm{O/H}$",
        "$\mathrm{S/H}$",
        "$\mathrm{C/O}$",
        "$\mathrm{S/O}$",
        "$\mathrm{M/H}$",
    ]
    name_file = ["C_H", "O_H", "S_H", "C_O", "S_O", "M_H"]

    for i, ratio in enumerate([c_h, o_h, s_h, c_o, s_o, m_h]):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hist(ratio, bins=20, color="grey")
        median = np.median(ratio)
        el, eu = hpdi(ratio, 0.68)
        ax.axvline(el, ls="--", color="black")
        ax.axvline(median, ls="--", color="black")
        ax.axvline(eu, ls="--", color="black")

        ax.set_xlabel(name[i])
        ax.set_ylabel("Probability Dencity")
        median_solar = median / 10 ** log_solar[i]
        median_solar_eu = eu / 10 ** log_solar[i] - median / 10 ** log_solar[i]
        median_solar_el = median / 10 ** log_solar[i] - el / 10 ** log_solar[i]
        ax.set_title(
            f"${median:.5f}^{{+{eu-median: .5f}}}_{{-{median-el:.5f}}}$ = "
            + rf"${median_solar:.5f}^{{+{median_solar_eu: .5f}}}_{{-{median_solar_el:.5f}}}\times$solar"
        )

        ax.tick_params(labelleft=False, left=False)

        plt.tight_layout()
        plt.savefig(dir_save + "ratio_" + name_file[i] + ".png")
        # plt.show()
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hist(np.log10(ratio), bins=20, color="grey")
        median = np.log10(np.median(ratio))
        el, eu = np.log10(hpdi(ratio, 0.68))
        ax.axvline(el, ls="--", color="black")
        ax.axvline(median, ls="--", color="black")
        ax.axvline(eu, ls="--", color="black")

        ax.set_xlabel("$\log$(" + name[i] + ")")
        ax.set_ylabel("Probability Dencity")
        ax.set_title(f"${median:.3f}^{{+{eu-median: .3f}}}_{{-{median-el:.3f}}}$")

        ax.tick_params(labelleft=False, left=False)

        plt.tight_layout()
        plt.savefig(dir_save + "ratio_log_" + name_file[i] + ".png")
        # plt.show()
        plt.close()
