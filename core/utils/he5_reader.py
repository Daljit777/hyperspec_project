"""
PRISMA HE5 file reader utilities.
Handles reading VNIR and SWIR cubes, wavelengths, and metadata.
"""
import h5py
import numpy as np
import json
import os


# PRISMA L1 HE5 internal paths
PRISMA_PATHS = {
    'l1': {
        'swath': 'HDFEOS/SWATHS/PRS_L1_HCO',
        'vnir_cube': 'HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/VNIR_Cube',
        'swir_cube': 'HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SWIR_Cube',
        'pan': 'HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/PANCRO',
        'lat': 'HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Latitude',
        'lon': 'HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Longitude',
    },
    'l2b': {
        'swath': 'HDFEOS/SWATHS/PRS_L2B_HCO',
        'vnir_cube': 'HDFEOS/SWATHS/PRS_L2B_HCO/Data Fields/VNIR_Cube',
        'swir_cube': 'HDFEOS/SWATHS/PRS_L2B_HCO/Data Fields/SWIR_Cube',
        'lat': 'HDFEOS/SWATHS/PRS_L2B_HCO/Geolocation Fields/Latitude',
        'lon': 'HDFEOS/SWATHS/PRS_L2B_HCO/Geolocation Fields/Longitude',
    },
    'l2c': {
        'swath': 'HDFEOS/SWATHS/PRS_L2C_HCO',
        'vnir_cube': 'HDFEOS/SWATHS/PRS_L2C_HCO/Data Fields/VNIR_Cube',
        'swir_cube': 'HDFEOS/SWATHS/PRS_L2C_HCO/Data Fields/SWIR_Cube',
        'lat': 'HDFEOS/SWATHS/PRS_L2C_HCO/Geolocation Fields/Latitude',
        'lon': 'HDFEOS/SWATHS/PRS_L2C_HCO/Geolocation Fields/Longitude',
    },
    'l2d': {
        'swath': 'HDFEOS/SWATHS/PRS_L2D_HCO',
        'vnir_cube': 'HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube',
        'swir_cube': 'HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube',
        'lat': 'HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Latitude',
        'lon': 'HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Longitude',
    },
}


def detect_prisma_level(h5_file):
    """Auto-detect PRISMA product level from HE5 structure."""
    with h5py.File(h5_file, 'r') as f:
        swaths = f.get('HDFEOS/SWATHS', {})
        if swaths is None:
            return 'l1'
        keys = list(swaths.keys()) if swaths else []
        for k in keys:
            k_low = k.lower()
            if 'l2d' in k_low:
                return 'l2d'
            elif 'l2c' in k_low:
                return 'l2c'
            elif 'l2b' in k_low:
                return 'l2b'
        return 'l1'


def explore_he5_structure(h5_path, max_depth=4):
    """Return a dict tree of the HE5 file structure for debugging."""
    structure = {}
    def _recurse(h5obj, d, depth):
        if depth > max_depth:
            return
        try:
            for key in h5obj.keys():
                item = h5obj[key]
                if hasattr(item, 'keys'):
                    d[key] = {}
                    _recurse(item, d[key], depth + 1)
                else:
                    try:
                        shape = item.shape
                        dtype = str(item.dtype)
                        d[key] = f"Dataset shape={shape} dtype={dtype}"
                    except Exception:
                        d[key] = "Dataset"
        except Exception:
            pass
    try:
        with h5py.File(h5_path, 'r') as f:
            _recurse(f, structure, 0)
    except Exception as e:
        structure['error'] = str(e)
    return structure


def get_prisma_wavelengths(h5_path, level=None):
    """
    Extract VNIR and SWIR wavelengths from PRISMA HE5 global attributes.
    Returns (vnir_wl, swir_wl) as lists of floats.
    """
    if level is None:
        level = detect_prisma_level(h5_path)

    vnir_wl = []
    swir_wl = []

    try:
        with h5py.File(h5_path, 'r') as f:
            attrs = dict(f.attrs)
            # Try global attributes first
            for k, v in attrs.items():
                kl = k.lower()
                if 'vnir' in kl and 'wavel' in kl:
                    arr = np.array(v).flatten()
                    vnir_wl = [float(x) for x in arr if x > 0]
                elif 'swir' in kl and 'wavel' in kl:
                    arr = np.array(v).flatten()
                    swir_wl = [float(x) for x in arr if x > 0]

            # Fallback: check HDFEOS INFORMATION group
            if not vnir_wl:
                info_path = 'HDFEOS INFORMATION'
                if info_path in f:
                    info_attrs = dict(f[info_path].attrs)
                    for k, v in info_attrs.items():
                        kl = k.lower()
                        if 'vnir' in kl and 'wavel' in kl:
                            arr = np.array(v).flatten()
                            vnir_wl = [float(x) for x in arr if x > 0]
                        elif 'swir' in kl and 'wavel' in kl:
                            arr = np.array(v).flatten()
                            swir_wl = [float(x) for x in arr if x > 0]

    except Exception as e:
        print(f"Warning: Could not read wavelengths: {e}")

    # PRISMA defaults if not found: VNIR 400-1010nm (66 bands), SWIR 920-2505nm (174 bands)
    if not vnir_wl:
        vnir_wl = list(np.linspace(400.3, 1010.6, 66).round(2))
    if not swir_wl:
        swir_wl = list(np.linspace(920.1, 2505.0, 174).round(2))

    return vnir_wl, swir_wl


def read_prisma_cube(h5_path, sensor='vnir', band_indices=None, row_slice=None, col_slice=None):
    """
    Read PRISMA VNIR or SWIR cube from HE5.
    Returns numpy array of shape (rows, cols, bands).
    PRISMA stores cube as (along_track, bands, across_track) = (rows, bands, cols).

    Parameters:
        h5_path: str path to HE5 file
        sensor: 'vnir' or 'swir'
        band_indices: list of band indices to load (None=all)
        row_slice: slice object for rows (None=all)
        col_slice: slice object for cols (None=all)
    """
    level = detect_prisma_level(h5_path)
    paths = PRISMA_PATHS[level]
    cube_key = 'vnir_cube' if sensor.lower() == 'vnir' else 'swir_cube'
    cube_path = paths[cube_key]

    try:
        with h5py.File(h5_path, 'r') as f:
            ds = f[cube_path]
            # PRISMA shape: (along_track, bands, across_track) = (rows, bands, cols)

            if band_indices is not None:
                # Select specific bands: ds[all_rows, specific_bands, all_cols]
                data = ds[:, band_indices, :]
            elif row_slice is not None or col_slice is not None:
                rs = row_slice or slice(None)
                cs = col_slice or slice(None)
                data = ds[rs, :, cs]
            else:
                data = ds[:]

            # Transpose from (rows, bands, cols) → (rows, cols, bands)
            if data.ndim == 3:
                data = np.transpose(data, (0, 2, 1))

            return data.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to read PRISMA {sensor.upper()} cube: {e}")


def read_prisma_band(h5_path, band_idx, sensor='vnir'):
    """Read a single band. Returns 2D array (rows, cols)."""
    level = detect_prisma_level(h5_path)
    paths = PRISMA_PATHS[level]
    cube_key = 'vnir_cube' if sensor.lower() == 'vnir' else 'swir_cube'
    cube_path = paths[cube_key]

    try:
        with h5py.File(h5_path, 'r') as f:
            ds = f[cube_path]
            # PRISMA shape: (rows, bands, cols)
            band = ds[:, band_idx, :]   # (rows, cols)
            return band.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to read band {band_idx}: {e}")


def get_prisma_metadata(h5_path):
    """Extract key metadata from PRISMA HE5 global attributes."""
    meta = {}
    try:
        with h5py.File(h5_path, 'r') as f:
            for k, v in f.attrs.items():
                try:
                    val = v
                    if isinstance(val, (np.ndarray,)):
                        if val.size < 10:
                            val = val.tolist()
                        else:
                            val = f"array shape={val.shape}"
                    elif isinstance(val, (bytes,)):
                        val = val.decode('utf-8', errors='replace')
                    elif isinstance(val, (np.integer,)):
                        val = int(val)
                    elif isinstance(val, (np.floating,)):
                        val = float(val)
                    meta[k] = val
                except Exception:
                    meta[k] = str(v)

            # Get cube dimensions
            level = detect_prisma_level(h5_path)
            paths = PRISMA_PATHS[level]
            try:
                vnir_ds = f[paths['vnir_cube']]
                # PRISMA shape: (along_track, bands, across_track) = (rows, bands, cols)
                meta['vnir_shape'] = list(vnir_ds.shape)
                meta['rows'] = vnir_ds.shape[0]
                meta['vnir_bands'] = vnir_ds.shape[1]
                meta['cols'] = vnir_ds.shape[2]
            except Exception:
                pass
            try:
                swir_ds = f[paths['swir_cube']]
                meta['swir_shape'] = list(swir_ds.shape)
                meta['swir_bands'] = swir_ds.shape[1]
            except Exception:
                pass
    except Exception as e:
        meta['error'] = str(e)
    return meta


def get_spectral_profile(h5_path, row, col, sensor='vnir'):
    """
    Extract spectral profile at pixel (row, col).
    Returns numpy array of shape (bands,).
    """
    level = detect_prisma_level(h5_path)
    paths = PRISMA_PATHS[level]
    cube_key = 'vnir_cube' if sensor.lower() == 'vnir' else 'swir_cube'
    cube_path = paths[cube_key]

    try:
        with h5py.File(h5_path, 'r') as f:
            ds = f[cube_path]
            # PRISMA shape: (rows, bands, cols) → pixel at [row, :, col]
            profile = ds[row, :, col]
            return profile.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to read spectral profile: {e}")


# ─── ENVI File Support ──────────────────────────────────────────────────────

def find_envi_header(path):
    """
    Given a path (which might be a .hdr or the binary file),
    find the matching .hdr file and binary file.
    Returns (hdr_path, img_path) or (None, None).
    """
    if path.lower().endswith('.hdr'):
        hdr_path = path
        # Binary file is same name without .hdr
        img_path = path[:-4]
        if not os.path.exists(img_path):
            # Try common ENVI binary extensions
            for ext in ['', '.img', '.dat', '.bsq', '.bip', '.bil']:
                test = path[:-4] + ext
                if os.path.exists(test):
                    img_path = test
                    break
        return hdr_path, img_path
    else:
        # path is the binary file, look for .hdr
        hdr_path = path + '.hdr'
        if os.path.exists(hdr_path):
            return hdr_path, path
        # Try stripping extension and adding .hdr
        base, ext = os.path.splitext(path)
        hdr_path = base + '.hdr'
        if os.path.exists(hdr_path):
            return hdr_path, path
        return None, path


def read_envi_cube(path, band_indices=None):
    """
    Read an ENVI format hyperspectral cube.
    Returns (rows, cols, bands) float32 array.
    """
    try:
        import spectral.io.envi as envi
    except ImportError:
        import spectral
        envi = spectral.io.envi

    hdr_path, img_path = find_envi_header(path)
    if hdr_path is None or not os.path.exists(hdr_path):
        raise RuntimeError(f"ENVI header not found for: {path}")

    img = envi.open(hdr_path, img_path)
    if band_indices is not None:
        data = np.array(img.read_bands(band_indices), dtype=np.float32)
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
    else:
        data = np.array(img.load(), dtype=np.float32)
    return data


def read_envi_band(path, band_idx):
    """Read a single band from an ENVI file. Returns 2D array (rows, cols)."""
    try:
        import spectral.io.envi as envi
    except ImportError:
        import spectral
        envi = spectral.io.envi

    hdr_path, img_path = find_envi_header(path)
    if hdr_path is None:
        raise RuntimeError(f"ENVI header not found for: {path}")

    img = envi.open(hdr_path, img_path)
    band = np.array(img.read_band(band_idx), dtype=np.float32)
    return band


def get_envi_metadata(path):
    """Extract metadata from an ENVI .hdr file."""
    try:
        import spectral.io.envi as envi
    except ImportError:
        import spectral
        envi = spectral.io.envi

    hdr_path, img_path = find_envi_header(path)
    if hdr_path is None:
        return {'error': f'Header not found for {path}'}

    img = envi.open(hdr_path, img_path)
    meta = {}
    meta['rows'] = img.nrows
    meta['cols'] = img.ncols
    meta['bands'] = img.nbands
    meta['interleave'] = getattr(img, 'interleave', 'unknown')
    meta['dtype'] = str(img.dtype) if hasattr(img, 'dtype') else 'unknown'

    # Get wavelengths from metadata
    hdr_meta = img.metadata if hasattr(img, 'metadata') else {}
    if 'wavelength' in hdr_meta:
        wl_list = hdr_meta['wavelength']
        meta['wavelengths'] = [float(w) for w in wl_list]
    if 'wavelength units' in hdr_meta:
        meta['wavelength_units'] = hdr_meta['wavelength units']
    if 'description' in hdr_meta:
        meta['description'] = hdr_meta['description']
    if 'sensor type' in hdr_meta:
        meta['sensor_type'] = hdr_meta['sensor type']
    if 'band names' in hdr_meta:
        meta['band_names'] = hdr_meta['band names'][:10]  # first 10

    return meta


def get_envi_spectral_profile(path, row, col):
    """Extract spectral profile at pixel (row, col) from ENVI file."""
    try:
        import spectral.io.envi as envi
    except ImportError:
        import spectral
        envi = spectral.io.envi

    hdr_path, img_path = find_envi_header(path)
    if hdr_path is None:
        raise RuntimeError(f"ENVI header not found for: {path}")

    img = envi.open(hdr_path, img_path)
    pixel = np.array(img.read_pixel(row, col), dtype=np.float32)
    return pixel

