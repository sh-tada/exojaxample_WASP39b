import numpy as np
from astropy.io import fits


def res_G395H(wavelength):
    """
    Returns spectral resolution of G395H at the input wavelength

    Parameters:
        wavelength (float): unit must be nm

    Returns:
        res (int or float): spectral resolution
    """
    hdul = fits.open("data/jwst_nirspec_g395h_disp.fits")

    # hdul.info()
    # print(" ")
    # print(repr(hdul[1].header))
    # print(" ")
    # print(repr(hdul[1].data))

    data_list = hdul[1].data
    data = []
    for i in data_list:
        data.append(list(i))
    data = np.array(data)
    # print(data)

    return np.interp(wavelength / 1000.0, data[:, 0], data[:, 2])


if __name__ == "__main__":
    wav = [2500.0, 3400.0, 4000.0, 4700.0, 5500.0]
    for i in wav:
        print(f"wavelength: {i} nm, resolution: {res_G395H(i)}")
