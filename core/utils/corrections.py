"""
Radiometric, Geometric, and Atmospheric correction utilities for PRISMA data.
Includes advanced methods: PRNU, Dark Current, Smear, Smile, Keystone,
GCP, Affine Georeferencing, Orthorectification, QUAC, 6S approximation.
"""
import numpy as np
import os


# ═══════════════════════════════════════════════════════════════════════════════
#  RADIOMETRIC CORRECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def apply_scale_offset(data, scale=1.0, offset=0.0):
    """Convert DN to radiance/reflectance: Physical = DN * scale + offset"""
    return data.astype(np.float32) * scale + offset


def dark_object_subtraction_band(band_data):
    """DOS on a single band — subtracts 1st percentile of non-zero values."""
    dark_val = np.percentile(band_data[band_data > 0], 1) if np.any(band_data > 0) else 0
    corrected = np.clip(band_data.astype(np.float32) - dark_val, 0, None)
    return corrected, float(dark_val)


def apply_gain_offset_correction(data, gain=None, offset=None):
    """
    Gain and Offset Correction: corrected = (DN - offset) / gain
    If not provided, estimates gain/offset from the data statistics.
    """
    result = data.astype(np.float32).copy()
    log = ["=== Gain & Offset Correction ==="]
    for b in range(data.shape[2]):
        band = result[:, :, b]
        g = gain if gain is not None else float(np.std(band[band > 0])) if np.any(band > 0) else 1.0
        o = offset if offset is not None else float(np.percentile(band[band > 0], 1)) if np.any(band > 0) else 0.0
        if g == 0:
            g = 1.0
        result[:, :, b] = (band - o) / g
    log.append(f"Applied gain={gain or 'auto'}, offset={offset or 'auto'} per band")
    return result, '\n'.join(log)


def apply_prnu_correction(data):
    """
    PRNU (Pixel Response Non-Uniformity) Correction.
    Normalizes each pixel's response by the column-mean to correct sensor
    non-uniformity across the detector array.
    """
    result = data.astype(np.float32).copy()
    log = ["=== PRNU Correction ==="]
    for b in range(data.shape[2]):
        band = result[:, :, b]
        col_mean = np.mean(band, axis=0, keepdims=True)
        col_mean[col_mean == 0] = 1.0
        global_mean = np.mean(band)
        result[:, :, b] = band * (global_mean / col_mean)
    log.append(f"Normalized {data.shape[2]} bands using column-mean response")
    return result, '\n'.join(log)


def apply_dark_current_correction(data, dark_frame=None):
    """
    Dark Current Correction. Subtracts the dark current frame.
    If no dark frame provided, estimates from the minimum across rows.
    """
    result = data.astype(np.float32).copy()
    log = ["=== Dark Current Correction ==="]
    for b in range(data.shape[2]):
        band = result[:, :, b]
        if dark_frame is not None:
            dc = dark_frame[:, :, b] if dark_frame.ndim == 3 else dark_frame
        else:
            dc = np.min(band, axis=0, keepdims=True)
        result[:, :, b] = np.clip(band - dc, 0, None)
    log.append(f"Subtracted dark current from {data.shape[2]} bands")
    avg_dc = float(np.mean(np.min(data, axis=0)))
    log.append(f"Mean estimated dark current: {avg_dc:.4f}")
    return result, '\n'.join(log)


def apply_smear_correction(data, integration_time=1.0):
    """
    Smear Correction for pushbroom sensors.
    Corrects the signal accumulated during CCD readout by estimating
    the smear contribution from column-averaged signal.
    """
    result = data.astype(np.float32).copy()
    log = ["=== Smear Correction ==="]
    rows, cols, bands = data.shape
    for b in range(bands):
        band = result[:, :, b]
        col_avg = np.mean(band, axis=0, keepdims=True)
        smear_estimate = col_avg / max(rows, 1)
        cumulative_smear = np.cumsum(np.ones((rows, 1)) * smear_estimate, axis=0)
        result[:, :, b] = np.clip(band - cumulative_smear * integration_time, 0, None)
    log.append(f"Smear corrected {bands} bands (integration_time={integration_time})")
    return result, '\n'.join(log)


def apply_smile_correction(data, wavelengths=None):
    """
    Smile Effect Correction.
    The smile effect causes wavelength shift across the detector columns.
    Corrects by interpolating each column's spectrum to the center-column wavelength.
    """
    result = data.astype(np.float32).copy()
    log = ["=== Smile Effect Correction ==="]
    rows, cols, bands = data.shape
    if wavelengths is None or len(wavelengths) < 2:
        log.append("No wavelengths provided; applying column-mean normalization fallback")
        for b in range(bands):
            col_means = np.mean(result[:, :, b], axis=0, keepdims=True)
            center_mean = np.mean(result[:, cols//2, b])
            ratio = center_mean / np.where(col_means > 0, col_means, 1.0)
            result[:, :, b] *= ratio
    else:
        wl = np.array(wavelengths, dtype=np.float32)
        center_col = cols // 2
        for r in range(rows):
            center_spec = result[r, center_col, :]
            for c in range(cols):
                if c == center_col:
                    continue
                pixel_spec = result[r, c, :]
                shift = np.mean(pixel_spec - center_spec)
                if abs(shift) > 1e-6:
                    correction = shift * (c - center_col) / cols
                    result[r, c, :] -= correction
    log.append(f"Smile correction applied across {cols} columns")
    return result, '\n'.join(log)


def apply_keystone_correction(data):
    """
    Keystone Effect Correction.
    Keystone causes spatial misregistration across spectral bands.
    Corrects by cross-correlation alignment of bands to a reference band.
    """
    from scipy.ndimage import shift as ndi_shift
    result = data.astype(np.float32).copy()
    log = ["=== Keystone Effect Correction ==="]
    rows, cols, bands = data.shape
    ref_band = bands // 2
    ref = result[:, :, ref_band]
    ref_col_profile = np.mean(ref, axis=0)
    shifts_applied = []
    for b in range(bands):
        if b == ref_band:
            shifts_applied.append(0.0)
            continue
        band = result[:, :, b]
        band_col_profile = np.mean(band, axis=0)
        corr = np.correlate(ref_col_profile - np.mean(ref_col_profile),
                           band_col_profile - np.mean(band_col_profile), mode='full')
        peak = np.argmax(corr) - (len(ref_col_profile) - 1)
        shift_val = float(peak)
        if abs(shift_val) > 0 and abs(shift_val) < cols * 0.1:
            for r in range(rows):
                result[r, :, b] = ndi_shift(band[r, :], -shift_val, mode='nearest')
            shifts_applied.append(shift_val)
        else:
            shifts_applied.append(0.0)
    log.append(f"Keystone correction: max shift = {max(abs(s) for s in shifts_applied):.2f} px")
    log.append(f"Reference band: {ref_band}")
    return result, '\n'.join(log)


def apply_radiometric_correction(data, method='scale_offset', scale=1.0, offset=0.0):
    """
    Master dispatcher for radiometric corrections.
    Methods: scale_offset, dos, normalize, percentile, gain_offset,
             prnu, dark_current, smear, smile, keystone
    """
    result = data.astype(np.float32).copy()
    log_lines = []

    if method == 'scale_offset':
        result = apply_scale_offset(result, scale, offset)
        log_lines.append(f"Applied scale={scale}, offset={offset}")
    elif method == 'dos':
        dark_vals = []
        for b in range(data.shape[2]):
            corrected, dark_val = dark_object_subtraction_band(result[:, :, b])
            result[:, :, b] = corrected
            dark_vals.append(dark_val)
        log_lines.append(f"DOS: mean dark value = {np.mean(dark_vals):.4f}")
    elif method == 'normalize':
        for b in range(data.shape[2]):
            band = result[:, :, b]
            bmin, bmax = band.min(), band.max()
            if bmax > bmin:
                result[:, :, b] = (band - bmin) / (bmax - bmin)
        log_lines.append("Band-wise normalization applied [0–1]")
    elif method == 'percentile':
        p2, p98 = np.percentile(result, 2), np.percentile(result, 98)
        result = np.clip(result, p2, p98)
        result = (result - p2) / (p98 - p2 + 1e-8)
        log_lines.append(f"Percentile stretch: p2={p2:.4f}, p98={p98:.4f}")
    elif method == 'gain_offset':
        result, log = apply_gain_offset_correction(result,
                                                    gain=scale if scale != 1.0 else None,
                                                    offset=offset if offset != 0.0 else None)
        return result, log
    elif method == 'prnu':
        return apply_prnu_correction(result)
    elif method == 'dark_current':
        return apply_dark_current_correction(result)
    elif method == 'smear':
        return apply_smear_correction(result)
    elif method == 'smile':
        return apply_smile_correction(result)
    elif method == 'keystone':
        return apply_keystone_correction(result)

    return result, '\n'.join(log_lines)


def compute_band_statistics(data):
    """Compute per-band statistics for a cube (rows, cols, bands)."""
    stats = []
    n_bands = data.shape[2] if data.ndim == 3 else 1
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    for b in range(n_bands):
        band = data[:, :, b]
        valid = band[~np.isnan(band)]
        if len(valid) == 0:
            stats.append({'band': b, 'min': 0, 'max': 0, 'mean': 0, 'std': 0})
        else:
            stats.append({
                'band': b, 'min': float(np.min(valid)), 'max': float(np.max(valid)),
                'mean': float(np.mean(valid)), 'std': float(np.std(valid)),
                'median': float(np.median(valid)),
            })
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
#  ATMOSPHERIC CORRECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def atmospheric_correction_dos(data, solar_zenith=30.0, wavelengths=None):
    """Dark Object Subtraction atmospheric correction."""
    result = data.astype(np.float32).copy()
    log_lines = ["=== Atmospheric Correction (DOS) ==="]
    path_radiances = []
    cos_theta = np.cos(np.radians(solar_zenith))
    log_lines.append(f"Solar zenith: {solar_zenith}°, cos(θ) = {cos_theta:.4f}")
    for b in range(data.shape[2]):
        band = result[:, :, b]
        valid = band[band > 0]
        dark_val = float(np.percentile(valid, 1)) if len(valid) > 0 else 0.0
        result[:, :, b] = np.clip(band - dark_val, 0, None)
        path_radiances.append(dark_val)
    log_lines.append(f"Mean path radiance subtracted: {np.mean(path_radiances):.4f}")
    return result, '\n'.join(log_lines)


def atmospheric_correction_empirical(data, target_reflectance=None, wavelengths=None):
    """Empirical Line Method — per-band 1%–99% stretch."""
    result = data.astype(np.float32).copy()
    log_lines = ["=== Atmospheric Correction (Empirical) ==="]
    for b in range(data.shape[2]):
        band = result[:, :, b]
        p1, p99 = float(np.percentile(band, 1)), float(np.percentile(band, 99))
        if p99 > p1:
            result[:, :, b] = np.clip((band - p1) / (p99 - p1), 0, 1)
    log_lines.append("Empirical line normalization applied.")
    return result, '\n'.join(log_lines)


def atmospheric_correction_quac(data, wavelengths=None):
    """
    QUAC (Quick Atmospheric Correction).
    Derives atmospheric compensation from scene statistics without
    requiring ground truth or atmospheric parameters.
    Uses the average endmember spectrum approach.
    """
    result = data.astype(np.float32).copy()
    log = ["=== QUAC (Quick Atmospheric Correction) ==="]
    rows, cols, bands = data.shape
    pixels = result.reshape(-1, bands)

    # Remove invalid pixels
    valid_mask = np.all(np.isfinite(pixels), axis=1) & np.any(pixels > 0, axis=1)
    valid_pixels = pixels[valid_mask]

    if len(valid_pixels) < 100:
        log.append("Not enough valid pixels for QUAC")
        return result, '\n'.join(log)

    # Step 1: Compute mean spectrum (proxy for average endmember)
    mean_spec = np.mean(valid_pixels, axis=0)
    log.append(f"Mean spectrum computed from {len(valid_pixels)} valid pixels")

    # Step 2: Estimate path radiance (dark object per band)
    path_rad = np.zeros(bands)
    for b in range(bands):
        v = valid_pixels[:, b]
        path_rad[b] = np.percentile(v, 1)

    # Step 3: Remove path radiance
    for b in range(bands):
        result[:, :, b] = np.clip(result[:, :, b] - path_rad[b], 0, None)

    # Step 4: Normalize by mean spectrum to approximate reflectance
    mean_after = np.mean(result.reshape(-1, bands)[valid_mask], axis=0)
    for b in range(bands):
        if mean_after[b] > 1e-6:
            result[:, :, b] = result[:, :, b] / mean_after[b]
        result[:, :, b] = np.clip(result[:, :, b], 0, 1.5)

    log.append(f"Path radiance removed (mean={np.mean(path_rad):.4f})")
    log.append("Scene-derived reflectance normalization applied")
    log.append(f"Output range: [{np.nanmin(result):.4f}, {np.nanmax(result):.4f}]")
    return result, '\n'.join(log)


def atmospheric_correction_6s(data, wavelengths=None, solar_zenith=30.0,
                                visibility=23.0, water_vapor=2.5, ozone=0.34):
    """
    Simplified 6S-based atmospheric correction.
    Approximates the 6S radiative transfer model using Rayleigh+aerosol
    scattering parameterization for each wavelength band.
    """
    result = data.astype(np.float32).copy()
    log = ["=== 6S Model-Based Atmospheric Correction ==="]
    rows, cols, bands = data.shape

    cos_sz = np.cos(np.radians(solar_zenith))
    log.append(f"Solar zenith={solar_zenith}°, visibility={visibility}km")
    log.append(f"Water vapor={water_vapor} g/cm², ozone={ozone} cm-atm")

    wl = np.array(wavelengths) if wavelengths else np.linspace(400, 2500, bands)

    for b in range(bands):
        wl_um = wl[b] / 1000.0 if wl[b] > 100 else wl[b]

        # Rayleigh optical depth (approximation)
        tau_r = 0.00864 * (wl_um ** (-3.916 + 0.074 * wl_um + 0.050 / wl_um))

        # Aerosol optical depth (Ångström relation)
        tau_a = (3.912 / visibility) * (wl_um / 0.55) ** (-1.3)

        # Water vapor absorption
        tau_w = 0.0
        if 900 < wl[b] < 980 or 1100 < wl[b] < 1200:
            tau_w = 0.1 * water_vapor
        elif 1350 < wl[b] < 1450:
            tau_w = 0.5 * water_vapor
        elif 1800 < wl[b] < 1950:
            tau_w = 0.8 * water_vapor

        # Ozone absorption (Chappuis band)
        tau_o = 0.0
        if 500 < wl[b] < 700:
            tau_o = 0.02 * ozone

        total_tau = tau_r + tau_a + tau_w + tau_o
        transmittance = np.exp(-total_tau / cos_sz)

        # Path radiance estimate
        band_data = result[:, :, b]
        valid = band_data[band_data > 0]
        Lp = np.percentile(valid, 0.5) if len(valid) > 0 else 0.0

        # Correct: reflectance ≈ (L - Lp) / (E_sun * T * cos(sz))
        E_sun = 1.0  # Normalized
        reflectance = (band_data - Lp) / (E_sun * max(transmittance, 0.01) * cos_sz)
        result[:, :, b] = np.clip(reflectance, 0, 1.5)

    log.append(f"Applied 6S correction to {bands} bands")
    log.append(f"Output range: [{np.nanmin(result):.4f}, {np.nanmax(result):.4f}]")
    return result, '\n'.join(log)


# ═══════════════════════════════════════════════════════════════════════════════
#  GEOMETRIC CORRECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def apply_geometric_correction_flip(data, flip_ud=False, flip_lr=False, rotate_90=0):
    """Basic geometric correction: flip and rotate."""
    result = data.copy()
    log_lines = ["=== Geometric Correction ==="]
    if flip_ud:
        result = result[::-1, :, :]
        log_lines.append("Flipped up-down")
    if flip_lr:
        result = result[:, ::-1, :]
        log_lines.append("Flipped left-right")
    for _ in range(rotate_90 % 4):
        result = np.rot90(result, k=1, axes=(0, 1))
        log_lines.append("Rotated 90°")
    return result, '\n'.join(log_lines)


def apply_gcp_correction(data, gcp_pairs=None):
    """
    GCP (Ground Control Point) based geometric correction.
    Uses control point pairs to compute a polynomial transformation.
    gcp_pairs: list of dicts with {src_row, src_col, dst_row, dst_col}
    """
    from scipy.ndimage import map_coordinates
    result = data.astype(np.float32).copy()
    log = ["=== GCP-Based Geometric Correction ==="]
    rows, cols, bands = data.shape

    if not gcp_pairs or len(gcp_pairs) < 3:
        log.append("Insufficient GCPs (need ≥3). Generating synthetic GCPs for demo.")
        gcp_pairs = [
            {'src_row': 0, 'src_col': 0, 'dst_row': 5, 'dst_col': 3},
            {'src_row': 0, 'src_col': cols-1, 'dst_row': 2, 'dst_col': cols-4},
            {'src_row': rows-1, 'src_col': 0, 'dst_row': rows-3, 'dst_col': 5},
            {'src_row': rows-1, 'src_col': cols-1, 'dst_row': rows-6, 'dst_col': cols-2},
            {'src_row': rows//2, 'src_col': cols//2, 'dst_row': rows//2+1, 'dst_col': cols//2+2},
        ]

    src_pts = np.array([[g['src_col'], g['src_row']] for g in gcp_pairs], dtype=np.float64)
    dst_pts = np.array([[g['dst_col'], g['dst_row']] for g in gcp_pairs], dtype=np.float64)

    # Compute affine from GCPs using least squares
    n = len(src_pts)
    A = np.zeros((2*n, 6))
    b_vec = np.zeros(2*n)
    for i in range(n):
        sx, sy = src_pts[i]
        dx, dy = dst_pts[i]
        A[2*i]   = [sx, sy, 1, 0, 0, 0]
        A[2*i+1] = [0, 0, 0, sx, sy, 1]
        b_vec[2*i] = dx
        b_vec[2*i+1] = dy

    params, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
    a, b_p, c, d, e, f = params

    # Apply inverse transform
    row_coords, col_coords = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    det = a * e - b_p * d
    if abs(det) < 1e-10:
        log.append("Degenerate transform — skipping")
        return result, '\n'.join(log)

    inv_col = (e * (col_coords - c) - b_p * (row_coords - f)) / det
    inv_row = (-d * (col_coords - c) + a * (row_coords - f)) / det

    for band_i in range(bands):
        result[:, :, band_i] = map_coordinates(
            data[:, :, band_i], [inv_row, inv_col], order=1, mode='nearest'
        )

    log.append(f"GCP correction applied with {len(gcp_pairs)} control points")
    log.append(f"Affine params: a={a:.4f}, b={b_p:.4f}, c={c:.4f}, d={d:.4f}, e={e:.4f}, f={f:.4f}")
    rmse = np.sqrt(np.mean((A @ params - b_vec)**2))
    log.append(f"RMSE: {rmse:.4f} pixels")
    return result, '\n'.join(log)


def apply_affine_georeferencing(data, scale_x=1.0, scale_y=1.0, rotation=0.0,
                                  translate_x=0.0, translate_y=0.0):
    """
    Direct Georeferencing using Affine Transformation.
    Applies scale, rotation, and translation to the image.
    """
    from scipy.ndimage import affine_transform
    result = data.astype(np.float32).copy()
    log = ["=== Affine Georeferencing ==="]
    rows, cols, bands = data.shape

    theta = np.radians(rotation)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    transform_matrix = np.array([
        [cos_t / scale_y, sin_t / scale_x],
        [-sin_t / scale_y, cos_t / scale_x]
    ])
    offset = np.array([
        rows/2 - transform_matrix[0,0]*rows/2 - transform_matrix[0,1]*cols/2 - translate_y,
        cols/2 - transform_matrix[1,0]*rows/2 - transform_matrix[1,1]*cols/2 - translate_x,
    ])

    for b in range(bands):
        result[:, :, b] = affine_transform(
            data[:, :, b], transform_matrix, offset=offset, order=1, mode='nearest'
        )

    log.append(f"Scale: ({scale_x:.3f}, {scale_y:.3f}), Rotation: {rotation:.1f}°")
    log.append(f"Translation: ({translate_x:.1f}, {translate_y:.1f}) pixels")
    return result, '\n'.join(log)


def apply_orthorectification(data, dem_scale=0.001):
    """
    Simplified Orthorectification.
    Simulates terrain relief correction using a synthetic DEM
    (estimated from band variance as a proxy for elevation variation).
    """
    from scipy.ndimage import map_coordinates
    result = data.astype(np.float32).copy()
    log = ["=== Orthorectification ==="]
    rows, cols, bands = data.shape

    # Generate proxy DEM from spectral variance (high variance ≈ terrain edges)
    band_var = np.var(data[:, :, :min(bands, 10)], axis=2)
    from scipy.ndimage import gaussian_filter
    dem = gaussian_filter(band_var, sigma=10)
    dem = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)

    # Compute displacement from DEM gradient
    dy, dx = np.gradient(dem)
    row_coords, col_coords = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    new_rows = row_coords - dy * dem_scale * rows
    new_cols = col_coords - dx * dem_scale * cols
    new_rows = np.clip(new_rows, 0, rows - 1)
    new_cols = np.clip(new_cols, 0, cols - 1)

    for b in range(bands):
        result[:, :, b] = map_coordinates(data[:, :, b], [new_rows, new_cols],
                                           order=1, mode='nearest')

    log.append(f"Orthorectification applied with DEM scale={dem_scale}")
    log.append(f"Proxy DEM range: [{dem.min():.4f}, {dem.max():.4f}]")
    log.append(f"Max displacement: {np.max(np.abs(dy)):.4f} rows, {np.max(np.abs(dx)):.4f} cols")
    return result, '\n'.join(log)


# ═══════════════════════════════════════════════════════════════════════════════
#  PCA
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pca(data, n_components=5):
    """PCA on hyperspectral cube (rows, cols, bands)."""
    from sklearn.decomposition import PCA as SkPCA
    rows, cols, bands = data.shape
    X = data.reshape(-1, bands).astype(np.float32)
    nan_mask = np.any(np.isnan(X), axis=1)
    X_clean = X[~nan_mask]
    n_components = min(n_components, bands, X_clean.shape[0])
    pca = SkPCA(n_components=n_components)
    pca.fit(X_clean)
    X_transformed = np.zeros((X.shape[0], n_components), dtype=np.float32)
    X_transformed[~nan_mask] = pca.transform(X_clean)
    pca_cube = X_transformed.reshape(rows, cols, n_components)
    return {
        'cube': pca_cube,
        'explained_variance': pca.explained_variance_ratio_.tolist(),
        'loadings': pca.components_.tolist(),
        'n_components': n_components,
    }
