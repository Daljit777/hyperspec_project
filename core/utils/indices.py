"""
Spectral indices computation for hyperspectral data.
Supports NDVI, EVI, NDWI, SAVI, MSAVI, NDMI, NBR and custom indices.
"""
import numpy as np


def get_band_by_wavelength(data, wavelengths, target_wl, tolerance=20.0):
    """
    Find the band index closest to target wavelength and return the 2D band.

    Parameters:
        data: (rows, cols, bands) array
        wavelengths: list of wavelengths for each band
        target_wl: target wavelength in nm
        tolerance: max allowed deviation in nm
    """
    wl_arr = np.array(wavelengths)
    idx = int(np.argmin(np.abs(wl_arr - target_wl)))
    actual_wl = wl_arr[idx]
    if abs(actual_wl - target_wl) > tolerance:
        return None, idx, actual_wl
    return data[:, :, idx].astype(np.float32), idx, actual_wl


def compute_ndvi(data, wavelengths):
    """
    Normalized Difference Vegetation Index.
    NDVI = (NIR - Red) / (NIR + Red)
    NIR ≈ 842 nm, Red ≈ 665 nm
    """
    nir, nir_idx, nir_wl = get_band_by_wavelength(data, wavelengths, 842, tolerance=40)
    red, red_idx, red_wl = get_band_by_wavelength(data, wavelengths, 665, tolerance=40)

    if nir is None or red is None:
        return None, f"Bands not found for NDVI (NIR≈{nir_wl}nm, Red≈{red_wl}nm)"

    denom = nir + red
    ndvi = np.where(np.abs(denom) > 1e-6, (nir - red) / denom, 0.0)
    ndvi = np.clip(ndvi, -1, 1)
    info = f"NDVI: NIR={nir_wl:.1f}nm (band {nir_idx}), Red={red_wl:.1f}nm (band {red_idx})"
    return ndvi.astype(np.float32), info


def compute_evi(data, wavelengths, G=2.5, C1=6.0, C2=7.5, L=1.0):
    """
    Enhanced Vegetation Index.
    EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)
    NIR≈842, Red≈665, Blue≈490
    """
    nir, ni, nwl = get_band_by_wavelength(data, wavelengths, 842, 40)
    red, ri, rwl = get_band_by_wavelength(data, wavelengths, 665, 40)
    blue, bi, bwl = get_band_by_wavelength(data, wavelengths, 490, 40)

    if any(x is None for x in [nir, red, blue]):
        return None, "Bands not found for EVI"

    denom = nir + C1 * red - C2 * blue + L
    evi = np.where(np.abs(denom) > 1e-6, G * (nir - red) / denom, 0.0)
    evi = np.clip(evi, -1, 1)
    info = f"EVI: NIR={nwl:.1f}nm, Red={rwl:.1f}nm, Blue={bwl:.1f}nm"
    return evi.astype(np.float32), info


def compute_ndwi(data, wavelengths):
    """
    Normalized Difference Water Index.
    NDWI = (Green - NIR) / (Green + NIR)
    Green ≈ 560 nm, NIR ≈ 842 nm
    """
    green, gi, gwl = get_band_by_wavelength(data, wavelengths, 560, 40)
    nir, ni, nwl = get_band_by_wavelength(data, wavelengths, 842, 40)

    if green is None or nir is None:
        return None, "Bands not found for NDWI"

    denom = green + nir
    ndwi = np.where(np.abs(denom) > 1e-6, (green - nir) / denom, 0.0)
    ndwi = np.clip(ndwi, -1, 1)
    info = f"NDWI: Green={gwl:.1f}nm, NIR={nwl:.1f}nm"
    return ndwi.astype(np.float32), info


def compute_savi(data, wavelengths, L=0.5):
    """
    Soil Adjusted Vegetation Index.
    SAVI = (NIR - Red) * (1 + L) / (NIR + Red + L)
    """
    nir, ni, nwl = get_band_by_wavelength(data, wavelengths, 842, 40)
    red, ri, rwl = get_band_by_wavelength(data, wavelengths, 665, 40)

    if nir is None or red is None:
        return None, "Bands not found for SAVI"

    denom = nir + red + L
    savi = np.where(np.abs(denom) > 1e-6, (nir - red) * (1 + L) / denom, 0.0)
    savi = np.clip(savi, -1, 1)
    info = f"SAVI (L={L}): NIR={nwl:.1f}nm, Red={rwl:.1f}nm"
    return savi.astype(np.float32), info


def compute_ndmi(data, wavelengths):
    """
    Normalized Difference Moisture Index.
    NDMI = (NIR - SWIR1) / (NIR + SWIR1)
    NIR ≈ 842 nm, SWIR1 ≈ 1609 nm
    """
    nir, ni, nwl = get_band_by_wavelength(data, wavelengths, 842, 40)
    swir, si, swl = get_band_by_wavelength(data, wavelengths, 1609, 60)

    if nir is None or swir is None:
        return None, "Bands not found for NDMI (may need SWIR data)"

    denom = nir + swir
    ndmi = np.where(np.abs(denom) > 1e-6, (nir - swir) / denom, 0.0)
    ndmi = np.clip(ndmi, -1, 1)
    info = f"NDMI: NIR={nwl:.1f}nm, SWIR={swl:.1f}nm"
    return ndmi.astype(np.float32), info


def compute_nbr(data, wavelengths):
    """
    Normalized Burn Ratio.
    NBR = (NIR - SWIR2) / (NIR + SWIR2)
    SWIR2 ≈ 2190 nm
    """
    nir, ni, nwl = get_band_by_wavelength(data, wavelengths, 842, 40)
    swir, si, swl = get_band_by_wavelength(data, wavelengths, 2190, 80)

    if nir is None or swir is None:
        return None, "Bands not found for NBR (may need SWIR data)"

    denom = nir + swir
    nbr = np.where(np.abs(denom) > 1e-6, (nir - swir) / denom, 0.0)
    nbr = np.clip(nbr, -1, 1)
    info = f"NBR: NIR={nwl:.1f}nm, SWIR={swl:.1f}nm"
    return nbr.astype(np.float32), info


def compute_custom_index(data, wavelengths, band1_wl, band2_wl, formula='ndiff'):
    """
    Compute a custom two-band index.
    formula: 'ndiff' = (b1-b2)/(b1+b2), 'ratio' = b1/b2, 'diff' = b1-b2
    """
    b1, i1, w1 = get_band_by_wavelength(data, wavelengths, band1_wl, 50)
    b2, i2, w2 = get_band_by_wavelength(data, wavelengths, band2_wl, 50)

    if b1 is None or b2 is None:
        return None, "One or both bands not found"

    if formula == 'ndiff':
        denom = b1 + b2
        result = np.where(np.abs(denom) > 1e-6, (b1 - b2) / denom, 0.0)
    elif formula == 'ratio':
        result = np.where(np.abs(b2) > 1e-6, b1 / b2, 0.0)
    else:  # diff
        result = b1 - b2

    info = f"Custom index ({formula}): B1={w1:.1f}nm, B2={w2:.1f}nm"
    return result.astype(np.float32), info


AVAILABLE_INDICES = {
    'ndvi': {'name': 'NDVI', 'desc': 'Vegetation (NIR-Red)/(NIR+Red)', 'fn': compute_ndvi},
    'evi': {'name': 'EVI', 'desc': 'Enhanced Vegetation Index', 'fn': compute_evi},
    'ndwi': {'name': 'NDWI', 'desc': 'Water Index (Green-NIR)/(Green+NIR)', 'fn': compute_ndwi},
    'savi': {'name': 'SAVI', 'desc': 'Soil Adjusted Vegetation Index', 'fn': compute_savi},
    'ndmi': {'name': 'NDMI', 'desc': 'Moisture Index (NIR-SWIR)/(NIR+SWIR)', 'fn': compute_ndmi},
    'nbr': {'name': 'NBR', 'desc': 'Normalized Burn Ratio', 'fn': compute_nbr},
}
