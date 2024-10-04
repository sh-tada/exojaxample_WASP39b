import ssl
import pickle
import jax.numpy as jnp
from jax import jit

from exojax.spec.contdb import CdbCIA
from exojax.spec.opacont import OpaCIA
from exojax.spec import molinfo

from exojax.spec.api import MdbHitemp
from exojax.spec.api import MdbExomol

from exojax.spec.opacalc import OpaPremodit

ssl._create_default_https_context = ssl._create_unverified_context


ciapath_list = {
    "H2H2": "/home/tada/.db_CIA/H2-H2_2011.cia",
    "H2He": "/home/tada/.db_CIA/H2-He_2011.cia",
}

db_HITEMP = "/home/tada/.db_HITEMP/"
db_ExoMol = "/home/tada/.db_ExoMol/"

# HITEMP isotope dict
HITEMP_iso_CO = {
    1: "(12C)(16O)",
    2: "(13C)(16O)",
    3: "(12C)(18O)",
    4: "(12C)(17O)",
    5: "(13C)(18O)",
    6: "(13C)(17O)",
}
HITEMP_iso_CO2 = {
    1: "(12C)(16O)2",
    2: "(13C)(16O)2",
    3: "(16O)(12C)(18O)",
    4: "(16O)(12C)(17O)",
    5: "(16O)(13C)(18O)",
    6: "(16O)(13C)(17O)",
    7: "(12C)(18O)2",
}
HITEMP_iso_H2O = {1: "H2(16O)", 2: "H2(18O)", 3: "H2(17O)"}
HITEMP_iso_CH4 = {1: "(12C)H4", 2: "(13C)H4", 3: "(12C)H3D"}
HITEMP_iso = {
    "CO": HITEMP_iso_CO,
    "CO2": HITEMP_iso_CO2,
    "H2O": HITEMP_iso_H2O,
    "CH4": HITEMP_iso_CH4,
}


molpath_list_HITEMP = {
    "(12C)(16O)": db_HITEMP + "CO/",
    # "(13C)(16O)": db_HITEMP + "CO/",
    # "(12C)(18O)": db_HITEMP + "CO/",
    "(12C)(16O)2": db_HITEMP + "CO2/",
    # "(13C)(16O)2": db_HITEMP + "CO2/",
    # "(16O)(12C)(18O)": db_HITEMP + "CO2/",
    "H2(16O)": db_HITEMP + "H2O/",
    # "H2(18O)": db_HITEMP + "H2O/",
    "(12C)H4": db_HITEMP + "CH4/",
}

molpath_list_Exomol = {
    "H2S": db_ExoMol + "H2S/1H2-32S/AYT2/",
    "SO2": db_ExoMol + "SO2/32S-16O2/ExoAmes/",
    "NH3": db_ExoMol + "NH3/14N-1H3/CoYuTe/",
    "HCN": db_ExoMol + "HCN/1H-12C-14N/Harris/",
    "C2H2": db_ExoMol + "C2H2/12C2-1H2/aCeTY/",
}


def opa_cias(nu_grid):
    """
    Returns a list of instances of class OpaCIA and a list of molecules

    Parameters:
        nu_grid (float): wavenumber grid

    Returns:
        ciamols (float): H2H2 or H2He
        opa_cias (OpaCIA): Opacity Continuum Calculator Class for CIA
    """
    ciamols = []
    opa_cias = []
    for ciamol in ciapath_list:
        ciamols.append(ciamol)
        cdb = CdbCIA(ciapath_list[ciamol], nurange=nu_grid)
        opa_cias.append(OpaCIA(cdb, nu_grid=nu_grid))
    return ciamols, opa_cias


def opa_cias_list(nu_grid_list):
    """
    Returns a list of a list of OpaCIA instances

    Parameters:
        nu_grid_list: List of wavenumber grids

    Returns:
        ciamols_list: List of cia molecules each wavenumber grid
        opa_cias_list: List of OpaCIA for each wavenumber grid
    """
    ciamols_list = []
    opa_cias_list = []
    for nu_grid in nu_grid_list:
        ciamols, opa_cias_ = opa_cias(nu_grid)
        ciamols_list.append(ciamols)
        opa_cias_list.append(opa_cias_)
    return ciamols_list[0], opa_cias_list


def opa_mols(nu_grid, Tlow, Thigh, diffmode=0):
    """
    Returns a list of instances of class OpaPremodit

    Parameters:
        nu_grid (float): wavenumber grid
        Tlow (float): Lower limit of the temperature range
        Thigh (float): Upper limit of the temperature range

    Returns:
        mols (int): List of molecule names
        opa_mols (OpaPremodit): Opacity Calculator Class for PreMODIT
        molmass_arr: jnp array of molecular mass
    """
    mols = []
    opas = []
    molmass_arr = []
    # HITEMP
    for mol in molpath_list_HITEMP:
        mols.append(mol)
        simple_mol = molpath_list_HITEMP[mol].rsplit("/")[-2]
        # number of isotope
        for isotope in HITEMP_iso[simple_mol]:
            if HITEMP_iso[simple_mol][isotope] == mol:
                break

        print(isotope, HITEMP_iso[simple_mol][isotope])
        mdb = MdbHitemp(
            molpath_list_HITEMP[mol], nu_grid, gpu_transfer=False, isotope=isotope
        )
        print()
        print(mdb.exact_isotope_name(isotope), f"molmass:{mdb.molmass}")
        molmass_arr.append(mdb.molmass)
        opas.append(
            OpaPremodit(
                mdb=mdb,
                nu_grid=nu_grid,
                diffmode=diffmode,
                auto_trange=[Tlow, Thigh],
                dit_grid_resolution=1,
                allow_32bit=True,
            )
        )

    # ExoMol
    for mol in molpath_list_Exomol:
        mols.append(mol)
        mdb = MdbExomol(molpath_list_Exomol[mol], nu_grid, gpu_transfer=False)
        molmass_arr.append(mdb.molmass)
        opas.append(
            OpaPremodit(
                mdb=mdb,
                nu_grid=nu_grid,
                diffmode=diffmode,
                auto_trange=[Tlow, Thigh],
                dit_grid_resolution=1,
                allow_32bit=True,
            )
        )

    return mols, opas, jnp.array(molmass_arr)


def opa_mols_list(nu_grid_list, Tlow, Thigh, diffmode=0, save_pickle=False):
    """
    Returns a list of OpaPremodit instances

    Parameters:
        nu_grid_list: List of wavenumber grids
        Tlow (float): Lower limit of the temperature range
        Thigh (float): Upper limit of the temperature range
        save_pickle: If Ture, opa_mol_list will be saved using pickle

    Returns:
        mols_list: List of molecular names for each wavenumber
        opa_mols_list: List of OpaPremodit for each wavenumber grid
        molmasses_list: List of molmass for each wavenumber
    """
    mols_list = []
    opa_mols_list = []
    molmass_arr_list = []
    for nu_grid in nu_grid_list:
        mols, opa_mols_, molmass_arr = opa_mols(nu_grid, Tlow, Thigh, diffmode)
        mols_list.append(mols)
        opa_mols_list.append(opa_mols_)
        molmass_arr_list.append(molmass_arr)

    if save_pickle:
        pickled_data = pickle.dumps(opa_mols_list)
        data_size = len(pickled_data)
        print(f"Data size: {data_size/1024/1024/1024} GiB")

        with open("OpaPremodit_dict.bin", "wb") as f:
            pickle.dump(opa_mols_list, f)
        with open("mols_list.bin", "wb") as f:
            pickle.dump(mols_list[0], f)
        with open("molmass_arr.bin", "wb") as f:
            pickle.dump(molmass_arr_list[0], f)

    return mols_list[0], opa_mols_list, molmass_arr_list[0]


def opa_mols_list_pickle():
    """
    Returns a list of OpaPremodit instances from pickle file

    Parameters:

    Returns:
        mols_list: List of molecular names for each wavenumber
        opa_mols_list: List of OpaPremodit for each wavenumber grid
        molmasses_list: List of molmass for each wavenumber
    """
    with open("OpaPremodit_dict.bin", "rb") as f:
        opa_mols_list = pickle.load(f)
    with open("mols_list.bin", "rb") as f:
        mols = pickle.load(f)
    with open("molmass_arr.bin", "rb") as f:
        molmass_arr = pickle.load(f)

    return mols, opa_mols_list, jnp.array(molmass_arr)


@jit
def vmrH2_vmrHe(vmr_arr):
    """
    Returns vmr of H2 and He at each layer

    Parameters:
        vmrs: jnp array of volume mixing ratio

    Returns:
        vmrH2: vmr of H2 array (n_layer)
        vmrHe: vmr of He array (n_layer)
    """
    vmr = jnp.sum(vmr_arr, axis=0)
    vmr = jnp.where(vmr >= 1.0, 1.0 - 10.0 ** (-10), vmr)
    vmrH2 = (1.0 - vmr) * 6.0 / 7.0
    vmrHe = (1.0 - vmr) * 1.0 / 7.0
    return vmrH2, vmrHe


@jit
def calc_mmw(vmr_arr, molmass_arr):
    """
    Returns mmw at each layer

    Parameters:
        vmrs: jnp array of volume mixing ratio
        molmasses: jnp array of molecular mass

    Returns:
        mmw_arr: mmw array (n_layer)
    """
    vmrH2, vmrHe = vmrH2_vmrHe(vmr_arr)
    mmw_arr = molinfo.molmass_isotope("H2") * vmrH2
    mmw_arr += molinfo.molmass_isotope("He", db_HIT=False) * vmrHe
    mmw_arr += jnp.dot(molmass_arr, vmr_arr)
    return mmw_arr


if __name__ == "__main__":
    pass
