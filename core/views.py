import os, json, base64, traceback
import numpy as np
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.conf import settings

from .models import DataFile, ProcessingJob, SpectralProfile
from .utils.he5_reader import (
    get_prisma_metadata, get_prisma_wavelengths,
    read_prisma_band, get_spectral_profile, explore_he5_structure,
    detect_prisma_level, read_envi_cube, read_envi_band,
    get_envi_metadata, get_envi_spectral_profile, find_envi_header
)
from .utils.visualization import (
    band_to_base64, rgb_composite_to_base64,
    spectral_profile_plotly_json, multi_profile_plotly_json,
    index_map_to_base64, band_histogram_plotly_json,
    band_stats_chart_plotly_json, pca_variance_plotly_json
)
from .utils.corrections import (
    apply_radiometric_correction, atmospheric_correction_dos,
    atmospheric_correction_empirical, atmospheric_correction_quac,
    atmospheric_correction_6s, apply_geometric_correction_flip,
    apply_gcp_correction, apply_affine_georeferencing,
    apply_orthorectification, compute_band_statistics
)
from .utils.indices import AVAILABLE_INDICES, compute_custom_index


# ── Auth ─────────────────────────────────────────────────────────────────────

def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        user = authenticate(request,
                            username=request.POST.get('username'),
                            password=request.POST.get('password'))
        if user:
            login(request, user)
            return redirect('dashboard')
        messages.error(request, 'Invalid credentials. Please try again.')
    return render(request, 'core/login.html')


def logout_view(request):
    logout(request)
    return redirect('login')


# ── Dashboard ─────────────────────────────────────────────────────────────────

@login_required
def dashboard(request):
    files = DataFile.objects.filter(user=request.user)
    jobs  = ProcessingJob.objects.filter(user=request.user)
    profiles = SpectralProfile.objects.filter(user=request.user)
    ctx = {
        'total_files': files.count(),
        'total_jobs':  jobs.count(),
        'done_jobs':   jobs.filter(status='done').count(),
        'total_profiles': profiles.count(),
        'recent_files': files[:6],
        'recent_jobs':  jobs[:5],
    }
    return render(request, 'core/dashboard.html', ctx)


# ── File Upload & List ────────────────────────────────────────────────────────

@login_required
def upload_file(request):
    if request.method == 'POST':
        uploaded = request.FILES.get('data_file')
        disk_path = request.POST.get('disk_path', '').strip()

        if not uploaded and not disk_path:
            messages.error(request, 'Please upload a file or provide a disk path.')
            return redirect('upload')

        try:
            df = DataFile(user=request.user)

            if uploaded:
                df.name = uploaded.name
                df.file = uploaded
                df.file_type = _detect_type(uploaded.name)
            else:
                if not os.path.exists(disk_path):
                    messages.error(request, f'Path not found: {disk_path}')
                    return redirect('upload')
                df.name = os.path.basename(disk_path)
                df.file_path = disk_path
                df.file_type = _detect_type(disk_path)

            df.save()
            _parse_metadata(df)
            messages.success(request, f'"{df.name}" loaded successfully!')
            return redirect('viewer', pk=df.pk)
        except Exception as e:
            messages.error(request, f'Error loading file: {e}')
            return redirect('upload')

    # Pre-populate disk path suggestions from data folder
    suggestions = _get_data_suggestions()
    return render(request, 'core/upload.html', {'suggestions': suggestions})


def _detect_type(name):
    n = name.lower()
    if n.endswith('.he5') or n.endswith('.h5') or n.endswith('.hdf5'):
        return 'he5'
    if n.endswith('.tif') or n.endswith('.tiff'):
        return 'tiff'
    if n.endswith('.hdr'):
        return 'envi'
    return 'envi'


def _get_data_suggestions():
    """Scan data/ folder for importable hyperspectral files."""
    data_dir = os.path.join(settings.BASE_DIR, 'data')
    suggestions = []
    if os.path.isdir(data_dir):
        # Scan top-level files
        for f in os.listdir(data_dir):
            fp = os.path.join(data_dir, f)
            if f.lower().endswith(('.he5', '.h5', '.tif', '.tiff', '.hdr')):
                suggestions.append({'name': f, 'path': fp,
                                    'size': os.path.getsize(fp)})
        # Scan subdirectories for ENVI .hdr files (AVIRIS)
        for d in os.listdir(data_dir):
            dp = os.path.join(data_dir, d)
            if os.path.isdir(dp):
                for f in os.listdir(dp):
                    if f.lower().endswith('.hdr'):
                        fp = os.path.join(dp, f)
                        suggestions.append({
                            'name': f'{d}/{f}',
                            'path': fp,
                            'size': os.path.getsize(fp),
                        })
    return suggestions


def _parse_metadata(df):
    """Parse and store metadata from HE5, TIFF, or ENVI file."""
    path = df.get_absolute_path()
    if not path or not os.path.exists(path):
        return
    try:
        if df.file_type == 'he5':
            meta = get_prisma_metadata(path)
            vnir_wl, swir_wl = get_prisma_wavelengths(path)
            df.rows = meta.get('rows')
            df.cols = meta.get('cols')
            df.vnir_bands = meta.get('vnir_bands')
            df.swir_bands = meta.get('swir_bands')
            df.bands = (df.vnir_bands or 0) + (df.swir_bands or 0)
            df.wavelengths_vnir = vnir_wl
            df.wavelengths_swir = swir_wl
            df.metadata = {k: v for k, v in meta.items()
                           if isinstance(v, (str, int, float, bool, list)) and k not in
                           ('vnir_shape', 'swir_shape')}
        elif df.file_type == 'tiff':
            import rasterio
            with rasterio.open(path) as src:
                df.rows = src.height
                df.cols = src.width
                df.bands = src.count
                df.metadata = {
                    'crs': str(src.crs),
                    'transform': list(src.transform),
                    'driver': src.driver,
                    'nodata': src.nodata,
                }
        elif df.file_type == 'envi':
            meta = get_envi_metadata(path)
            df.rows = meta.get('rows')
            df.cols = meta.get('cols')
            df.bands = meta.get('bands')
            df.vnir_bands = meta.get('bands')  # treat all as VNIR for ENVI
            wls = meta.get('wavelengths', [])
            if wls:
                df.wavelengths_vnir = wls
            df.metadata = {k: v for k, v in meta.items()
                           if isinstance(v, (str, int, float, bool, list))
                           and k != 'wavelengths'}
        df.status = 'ready'
        df.save()
    except Exception as e:
        df.status = 'error'
        df.notes = str(e)
        df.save()


@login_required
def file_list(request):
    files = DataFile.objects.filter(user=request.user)
    return render(request, 'core/file_list.html', {'files': files})


@login_required
def delete_file(request, pk):
    df = get_object_or_404(DataFile, pk=pk, user=request.user)
    df.delete()
    messages.success(request, 'File deleted.')
    return redirect('file_list')


# ── Viewer ────────────────────────────────────────────────────────────────────

@login_required
def viewer(request, pk):
    df = get_object_or_404(DataFile, pk=pk, user=request.user)
    saved_profiles = SpectralProfile.objects.filter(data_file=df, user=request.user)

    vnir_wl = df.wavelengths_vnir or []
    swir_wl = df.wavelengths_swir or []
    n_vnir = df.vnir_bands or len(vnir_wl) or 66
    n_swir = df.swir_bands or len(swir_wl) or 174

    ctx = {
        'df': df,
        'active_file': df,
        'vnir_bands': n_vnir,
        'swir_bands': n_swir,
        'wavelengths_vnir': json.dumps(vnir_wl),
        'wavelengths_swir': json.dumps(swir_wl),
        'saved_profiles': saved_profiles,
    }
    return render(request, 'core/viewer.html', ctx)


# ── Corrections ───────────────────────────────────────────────────────────────

@login_required
def corrections_view(request, pk):
    df = get_object_or_404(DataFile, pk=pk, user=request.user)
    jobs = ProcessingJob.objects.filter(data_file=df, user=request.user,
                                        job_type__in=['radiometric','atmospheric','geometric'])
    return render(request, 'core/corrections.html', {'df': df, 'jobs': jobs, 'active_file': df})


# ── Analysis ──────────────────────────────────────────────────────────────────

@login_required
def analysis_view(request, pk):
    df = get_object_or_404(DataFile, pk=pk, user=request.user)
    indices_info = {k: {'name': v['name'], 'desc': v['desc']} for k, v in AVAILABLE_INDICES.items()}
    jobs = ProcessingJob.objects.filter(data_file=df, user=request.user)
    return render(request, 'core/analysis.html', {
        'df': df,
        'active_file': df,
        'indices': indices_info,
        'jobs': jobs,
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: Read data cube regardless of file type
# ═══════════════════════════════════════════════════════════════════════════════

def _read_cube(df, path, sensor='vnir', band_indices=None):
    """Read cube data from any file type. Returns (rows, cols, bands) array."""
    if df.file_type == 'he5':
        from .utils.he5_reader import read_prisma_cube
        return read_prisma_cube(path, sensor=sensor, band_indices=band_indices)
    elif df.file_type == 'envi':
        return read_envi_cube(path, band_indices=band_indices)
    else:
        import rasterio
        with rasterio.open(path) as src:
            if band_indices is not None:
                bands = [src.read(b+1).astype(np.float32) for b in band_indices]
                return np.stack(bands, axis=-1)
            else:
                return np.stack([src.read(b+1) for b in range(src.count)], axis=-1).astype(np.float32)


def _read_single_band(df, path, band_idx, sensor='vnir'):
    """Read a single band from any file type. Returns 2D array."""
    if df.file_type == 'he5':
        return read_prisma_band(path, band_idx, sensor)
    elif df.file_type == 'envi':
        return read_envi_band(path, band_idx)
    else:
        import rasterio
        with rasterio.open(path) as src:
            return src.read(min(band_idx + 1, src.count)).astype(np.float32)


def _get_wavelengths(df, sensor='vnir'):
    """Get wavelengths list for a file."""
    if df.file_type == 'he5':
        return df.wavelengths_vnir if sensor == 'vnir' else df.wavelengths_swir
    elif df.file_type == 'envi':
        return df.wavelengths_vnir  # ENVI stores all wavelengths in vnir field
    return None


def _read_cube_subsampled(df, path, sensor='vnir', max_pixels=500000):
    """Read cube with spatial subsampling if too large for memory.
    Limits total pixels to max_pixels (default 500k ≈ 700x700).
    Returns (data, scale_factor) where scale_factor is the step used.
    """
    if df.file_type == 'he5':
        from .utils.he5_reader import read_prisma_cube
        data = read_prisma_cube(path, sensor=sensor)
        rows, cols, bands = data.shape
    elif df.file_type == 'envi':
        # For ENVI, check shape first before loading
        from spectral import open_image
        from .utils.he5_reader import find_envi_header
        hdr_path = find_envi_header(path)
        img = open_image(hdr_path)
        rows, cols, bands = img.shape
        total = rows * cols
        if total > max_pixels:
            step = int(np.ceil(np.sqrt(total / max_pixels)))
            row_idx = list(range(0, rows, step))
            col_idx = list(range(0, cols, step))
            data = np.array(img.read_subimage(row_idx, col_idx), dtype=np.float32)
            return data, step
        data = np.array(img.load(), dtype=np.float32)
        return data, 1
    else:
        import rasterio
        with rasterio.open(path) as src:
            data = np.stack([src.read(b+1) for b in range(src.count)], axis=-1).astype(np.float32)
            rows, cols, bands = data.shape

    # Subsample if needed
    total = rows * cols
    if total > max_pixels:
        step = int(np.ceil(np.sqrt(total / max_pixels)))
        data = data[::step, ::step, :]
        return data, step
    return data, 1


# ═══════════════════════════════════════════════════════════════════════════════
#  API ENDPOINTS (AJAX / JSON)
# ═══════════════════════════════════════════════════════════════════════════════

@login_required
def api_band_image(request, pk):
    """Return base64 PNG of a single band."""
    df = get_object_or_404(DataFile, pk=pk, user=request.user)
    band_idx = int(request.GET.get('band', 0))
    sensor   = request.GET.get('sensor', 'vnir')
    cmap     = request.GET.get('cmap', 'inferno')
    path     = df.get_absolute_path()
    if not path:
        return JsonResponse({'error': 'No file path'}, status=400)
    try:
        band_data = _read_single_band(df, path, band_idx, sensor)
        wls = _get_wavelengths(df, sensor)
        wl = wls[band_idx] if wls and band_idx < len(wls) else None
        img_b64 = band_to_base64(band_data, colormap=cmap,
                                  title=f'{sensor.upper()} Band {band_idx}', wavelength=wl)
        return JsonResponse({'image': img_b64, 'shape': list(band_data.shape),
                             'wavelength': wl})
    except Exception as e:
        return JsonResponse({'error': str(e), 'trace': traceback.format_exc()}, status=500)


@login_required
def api_rgb_image(request, pk):
    """Return base64 PNG of RGB composite."""
    df = get_object_or_404(DataFile, pk=pk, user=request.user)
    r = int(request.GET.get('r', 29))
    g = int(request.GET.get('g', 19))
    b = int(request.GET.get('b', 9))
    sensor = request.GET.get('sensor', 'vnir')
    path = df.get_absolute_path()
    if not path:
        return JsonResponse({'error': 'No file path'}, status=400)
    try:
        # Clamp band indices to valid range
        if df.file_type == 'he5':
            max_band = (df.vnir_bands or 66) - 1 if sensor == 'vnir' else (df.swir_bands or 174) - 1
        else:
            max_band = (df.bands or 100) - 1
        r = min(r, max_band)
        g = min(g, max_band)
        b = min(b, max_band)

        data = _read_cube(df, path, sensor=sensor, band_indices=[r, g, b])
        img_b64 = rgb_composite_to_base64(data, 0, 1, 2,
                                          title=f'{sensor.upper()} RGB (B{r},B{g},B{b})')
        return JsonResponse({'image': img_b64})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def api_spectral_profile(request, pk):
    """Return spectral profile JSON for pixel (row, col)."""
    df = get_object_or_404(DataFile, pk=pk, user=request.user)
    row    = int(request.GET.get('row', 0))
    col    = int(request.GET.get('col', 0))
    sensor = request.GET.get('sensor', 'vnir')
    path   = df.get_absolute_path()
    if not path:
        return JsonResponse({'error': 'No file path'}, status=400)
    try:
        if df.file_type == 'he5':
            spectrum = get_spectral_profile(path, row, col, sensor)
        elif df.file_type == 'envi':
            spectrum = get_envi_spectral_profile(path, row, col)
        else:
            import rasterio
            with rasterio.open(path) as src:
                r_clamp = max(0, min(row, src.height - 1))
                c_clamp = max(0, min(col, src.width - 1))
                spectrum = np.array([src.read(b+1)[r_clamp, c_clamp] for b in range(src.count)],
                                    dtype=np.float32)

        wls = _get_wavelengths(df, sensor)
        wl_list = list(wls) if wls else list(range(len(spectrum)))
        chart_json = spectral_profile_plotly_json(
            spectrum.tolist(), wl_list, label=f'({row},{col})', row=row, col=col)
        return JsonResponse({
            'spectrum': spectrum.tolist(),
            'wavelengths': wl_list,
            'chart': chart_json,
            'row': row, 'col': col, 'sensor': sensor,
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def api_save_profile(request, pk):
    """Save a spectral profile to DB."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    df = get_object_or_404(DataFile, pk=pk, user=request.user)
    data = json.loads(request.body)
    sp = SpectralProfile.objects.create(
        data_file=df, user=request.user,
        label=data.get('label', 'Profile'),
        row=data['row'], col=data['col'],
        spectrum=data['spectrum'],
        wavelengths=data['wavelengths'],
        sensor_type=data.get('sensor', 'VNIR').upper(),
    )
    return JsonResponse({'id': sp.pk, 'label': sp.label})


@login_required
def api_run_correction(request, pk):
    """Run a correction job and return before/after images."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    df = get_object_or_404(DataFile, pk=pk, user=request.user)
    body    = json.loads(request.body)
    corr    = body.get('correction', 'radiometric')
    method  = body.get('method', 'dos')
    sensor  = body.get('sensor', 'vnir')
    band_idx = int(body.get('band_idx', 10))
    path    = df.get_absolute_path()
    if not path:
        return JsonResponse({'error': 'No file'}, status=400)

    job = ProcessingJob.objects.create(
        data_file=df, user=request.user,
        job_type=corr, status='running',
        parameters=body,
    )

    try:
        # Read a subset for preview (avoid loading full cube in memory)
        b_indices = list(range(max(0, band_idx-2), band_idx+3))
        subset = _read_cube(df, path, sensor=sensor, band_indices=b_indices)
        center = min(2, subset.shape[2]-1)
        before_band = subset[:, :, center]
        wls = _get_wavelengths(df, sensor)
        wl = wls[band_idx] if wls and band_idx < len(wls) else None

        before_img = band_to_base64(before_band, title='Before Correction', wavelength=wl)

        # Apply correction
        log_text = ''
        if corr == 'radiometric':
            scale  = float(body.get('scale', 1.0))
            offset = float(body.get('offset', 0.0))
            corrected, log_text = apply_radiometric_correction(subset, method, scale, offset)
        elif corr == 'atmospheric':
            sz = float(body.get('solar_zenith', 30.0))
            wls = _get_wavelengths(df, sensor)
            if method == 'dos':
                corrected, log_text = atmospheric_correction_dos(subset, sz)
            elif method == 'quac':
                corrected, log_text = atmospheric_correction_quac(subset, wls)
            elif method == '6s':
                vis = float(body.get('visibility', 23.0))
                wv  = float(body.get('water_vapor', 2.5))
                oz  = float(body.get('ozone', 0.34))
                corrected, log_text = atmospheric_correction_6s(subset, wls, sz, vis, wv, oz)
            else:
                corrected, log_text = atmospheric_correction_empirical(subset)
        elif corr == 'geometric':
            if method == 'gcp':
                gcp_pairs = body.get('gcp_pairs', None)
                corrected, log_text = apply_gcp_correction(subset, gcp_pairs)
            elif method == 'affine':
                sx = float(body.get('scale_x', 1.0))
                sy = float(body.get('scale_y', 1.0))
                rot = float(body.get('rotation', 0.0))
                tx = float(body.get('translate_x', 0.0))
                ty = float(body.get('translate_y', 0.0))
                corrected, log_text = apply_affine_georeferencing(subset, sx, sy, rot, tx, ty)
            elif method == 'ortho':
                ds = float(body.get('dem_scale', 0.001))
                corrected, log_text = apply_orthorectification(subset, ds)
            else:
                flip_ud = body.get('flip_ud', False)
                flip_lr = body.get('flip_lr', False)
                rot     = int(body.get('rotate_90', 0))
                corrected, log_text = apply_geometric_correction_flip(subset, flip_ud, flip_lr, rot)
        else:
            corrected = subset
            log_text = 'No correction applied.'

        after_band = corrected[:, :, min(center, corrected.shape[2]-1)] if corrected.ndim == 3 else corrected
        after_img  = band_to_base64(after_band, title='After Correction', wavelength=wl)

        hist_before = band_histogram_plotly_json(before_band, title='Histogram Before')
        hist_after  = band_histogram_plotly_json(after_band,  title='Histogram After',
                                                  color='#39ff14')

        job.status = 'done'
        job.log    = log_text
        job.completed_at = timezone.now()
        job.save()

        return JsonResponse({
            'before': before_img,
            'after':  after_img,
            'hist_before': hist_before,
            'hist_after':  hist_after,
            'log': log_text,
            'job_id': job.pk,
        })
    except Exception as e:
        job.status = 'error'
        job.log    = traceback.format_exc()
        job.save()
        return JsonResponse({'error': str(e), 'trace': traceback.format_exc()}, status=500)


@login_required
def api_compute_index(request, pk):
    """Compute a spectral index and return colormap image."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    df = get_object_or_404(DataFile, pk=pk, user=request.user)
    body   = json.loads(request.body)
    index_key = body.get('index', 'ndvi')
    sensor = body.get('sensor', 'vnir')
    path   = df.get_absolute_path()
    if not path:
        return JsonResponse({'error': 'No file'}, status=400)

    job = ProcessingJob.objects.create(
        data_file=df, user=request.user,
        job_type='index', status='running', parameters=body)

    try:
        data, _ = _read_cube_subsampled(df, path, sensor=sensor)
        wls = _get_wavelengths(df, sensor)
        if wls is None:
            wls = list(range(data.shape[2]))

        if index_key == 'custom':
            b1 = float(body.get('band1_wl', 842))
            b2 = float(body.get('band2_wl', 665))
            formula = body.get('formula', 'ndiff')
            idx_arr, info = compute_custom_index(data, wls, b1, b2, formula)
        else:
            fn = AVAILABLE_INDICES.get(index_key, {}).get('fn')
            if fn is None:
                return JsonResponse({'error': 'Unknown index'}, status=400)
            idx_arr, info = fn(data, wls)

        if idx_arr is None:
            job.status = 'error'; job.log = info; job.save()
            return JsonResponse({'error': info}, status=400)

        # Stats
        valid = idx_arr[np.isfinite(idx_arr)]
        stats = {
            'min': float(valid.min()), 'max': float(valid.max()),
            'mean': float(valid.mean()), 'std': float(valid.std()),
        }

        cmap_map = {'ndvi': 'RdYlGn', 'evi': 'RdYlGn', 'ndwi': 'RdBu',
                    'savi': 'RdYlGn', 'ndmi': 'Blues', 'nbr': 'RdYlBu'}
        cmap = cmap_map.get(index_key, 'RdYlGn')
        img_b64 = index_map_to_base64(idx_arr, title=index_key.upper(), colormap=cmap)

        job.status = 'done'; job.log = info
        job.result_data = stats; job.completed_at = timezone.now(); job.save()

        return JsonResponse({'image': img_b64, 'info': info, 'stats': stats, 'job_id': job.pk})
    except Exception as e:
        job.status = 'error'; job.log = traceback.format_exc(); job.save()
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def api_band_stats(request, pk):
    """Return band statistics as Plotly JSON."""
    df = get_object_or_404(DataFile, pk=pk, user=request.user)
    sensor = request.GET.get('sensor', 'vnir')
    path   = df.get_absolute_path()
    if not path:
        return JsonResponse({'error': 'No file'}, status=400)
    try:
        data, _ = _read_cube_subsampled(df, path, sensor=sensor)
        wls = _get_wavelengths(df, sensor)
        stats = compute_band_statistics(data)
        chart = band_stats_chart_plotly_json(stats, wls)
        return JsonResponse({'stats': stats, 'chart': chart})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def api_pca(request, pk):
    """Run PCA and return component images + variance chart."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    df = get_object_or_404(DataFile, pk=pk, user=request.user)
    body   = json.loads(request.body)
    sensor = body.get('sensor', 'vnir')
    n_comp = int(body.get('n_components', 5))
    path   = df.get_absolute_path()
    if not path:
        return JsonResponse({'error': 'No file'}, status=400)

    job = ProcessingJob.objects.create(
        data_file=df, user=request.user,
        job_type='pca', status='running', parameters=body)
    try:
        data, _ = _read_cube_subsampled(df, path, sensor=sensor)

        from .utils.corrections import compute_pca
        result = compute_pca(data, n_components=n_comp)

        # Build PC images
        pc_images = []
        for i in range(result['n_components']):
            img = band_to_base64(result['cube'][:, :, i],
                                  colormap='plasma',
                                  title=f'PC {i+1} ({result["explained_variance"][i]*100:.1f}%)')
            pc_images.append(img)

        var_chart = pca_variance_plotly_json(result['explained_variance'])

        job.status = 'done'
        job.result_data = {'explained_variance': result['explained_variance']}
        job.completed_at = timezone.now()
        job.save()

        return JsonResponse({
            'pc_images': pc_images,
            'variance_chart': var_chart,
            'explained_variance': result['explained_variance'],
        })
    except Exception as e:
        job.status = 'error'; job.log = traceback.format_exc(); job.save()
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def api_file_structure(request, pk):
    """Return HE5 file internal structure for debugging."""
    df = get_object_or_404(DataFile, pk=pk, user=request.user)
    path = df.get_absolute_path()
    if not path or df.file_type != 'he5':
        return JsonResponse({'error': 'HE5 file required'}, status=400)
    structure = explore_he5_structure(path)
    return JsonResponse({'structure': structure})


@login_required
def api_profiles_list(request, pk):
    """Return all saved profiles for a file as multi-profile chart."""
    df = get_object_or_404(DataFile, pk=pk, user=request.user)
    profiles = SpectralProfile.objects.filter(data_file=df, user=request.user)
    if not profiles.exists():
        return JsonResponse({'chart': None, 'profiles': []})
    prof_list = [{'spectrum': p.spectrum, 'wavelengths': p.wavelengths,
                  'label': p.label} for p in profiles]
    chart = multi_profile_plotly_json(prof_list)
    return JsonResponse({'chart': chart, 'profiles': list(
        profiles.values('pk', 'label', 'row', 'col', 'sensor_type', 'created_at'))})


# ── Advanced Analysis API ─────────────────────────────────────────────────────

@login_required
@csrf_exempt
def api_run_ppi(request, pk):
    """Run PPI endmember extraction."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    df = get_object_or_404(DataFile, pk=pk, user=request.user)
    body = json.loads(request.body)
    sensor = body.get('sensor', 'vnir')
    n_iter = int(body.get('n_iterations', 500))
    path = df.get_absolute_path()

    job = ProcessingJob.objects.create(
        data_file=df, user=request.user, job_type='ppi',
        status='running', parameters=body)
    try:
        from .utils.analysis import compute_ppi
        data, _ = _read_cube_subsampled(df, path, sensor=sensor)
        ppi_map, endmembers, info = compute_ppi(data, n_iterations=n_iter)

        ppi_img = band_to_base64(ppi_map, colormap='hot', title='PPI Scores')

        # Endmember spectral chart
        wls = _get_wavelengths(df, sensor) or list(range(data.shape[2]))
        em_list = [{'spectrum': em.tolist(), 'wavelengths': wls,
                    'label': f'EM-{i+1}'} for i, em in enumerate(endmembers)]
        em_chart = multi_profile_plotly_json(em_list)

        job.status = 'done'
        job.log = info
        job.result_data = {'n_endmembers': len(endmembers),
                          'endmembers': [em.tolist() for em in endmembers]}
        job.completed_at = timezone.now()
        job.save()

        return JsonResponse({
            'ppi_image': ppi_img, 'endmember_chart': em_chart,
            'n_endmembers': len(endmembers), 'info': info,
            'endmembers': [em.tolist() for em in endmembers],
        })
    except Exception as e:
        job.status = 'error'; job.log = traceback.format_exc(); job.save()
        return JsonResponse({'error': str(e)}, status=500)


@login_required
@csrf_exempt
def api_run_sam(request, pk):
    """Run SAM target detection."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    df = get_object_or_404(DataFile, pk=pk, user=request.user)
    body = json.loads(request.body)
    sensor = body.get('sensor', 'vnir')
    path = df.get_absolute_path()

    job = ProcessingJob.objects.create(
        data_file=df, user=request.user, job_type='sam',
        status='running', parameters=body)
    try:
        from .utils.analysis import compute_sam, compute_sam_classification, compute_ppi
        data, _ = _read_cube_subsampled(df, path, sensor=sensor)

        # Get endmembers: from body or PPI
        endmembers = body.get('endmembers', None)
        if not endmembers:
            _, endmembers_arr, _ = compute_ppi(data, n_iterations=300)
            endmembers = [em.tolist() for em in endmembers_arr]

        class_map, labels, info = compute_sam_classification(
            data, endmembers, labels=None)

        class_img = band_to_base64(class_map.astype(np.float32),
                                    colormap='tab10', title='SAM Classification')

        # Also compute angle for first endmember
        sam_map, sam_info = compute_sam(data, endmembers[0])
        angle_img = band_to_base64(sam_map, colormap='viridis_r',
                                    title='SAM Angle (EM-1)')

        job.status = 'done'; job.log = info
        job.completed_at = timezone.now(); job.save()

        return JsonResponse({
            'class_image': class_img, 'angle_image': angle_img,
            'info': info, 'n_classes': len(endmembers),
            'labels': labels,
        })
    except Exception as e:
        job.status = 'error'; job.log = traceback.format_exc(); job.save()
        return JsonResponse({'error': str(e)}, status=500)


@login_required
@csrf_exempt
def api_run_lsu(request, pk):
    """Run Linear Spectral Unmixing."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    df = get_object_or_404(DataFile, pk=pk, user=request.user)
    body = json.loads(request.body)
    sensor = body.get('sensor', 'vnir')
    path = df.get_absolute_path()

    job = ProcessingJob.objects.create(
        data_file=df, user=request.user, job_type='lsu',
        status='running', parameters=body)
    try:
        from .utils.analysis import compute_lsu, compute_ppi
        data, _ = _read_cube_subsampled(df, path, sensor=sensor)

        endmembers = body.get('endmembers', None)
        if not endmembers:
            _, em_arr, _ = compute_ppi(data, n_iterations=300)
            endmembers = [em.tolist() for em in em_arr]

        abundance_cube, info = compute_lsu(data, endmembers)

        # Generate abundance maps
        ab_images = []
        for i in range(abundance_cube.shape[2]):
            img = band_to_base64(abundance_cube[:, :, i], colormap='YlOrRd',
                                  title=f'EM-{i+1} Abundance')
            ab_images.append(img)

        job.status = 'done'; job.log = info
        job.completed_at = timezone.now(); job.save()

        return JsonResponse({
            'abundance_images': ab_images[:6],
            'info': info, 'n_endmembers': len(endmembers),
        })
    except Exception as e:
        job.status = 'error'; job.log = traceback.format_exc(); job.save()
        return JsonResponse({'error': str(e)}, status=500)


@login_required
@csrf_exempt
def api_run_classification(request, pk):
    """Run supervised classification (SVM or Random Forest)."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    df = get_object_or_404(DataFile, pk=pk, user=request.user)
    body = json.loads(request.body)
    sensor = body.get('sensor', 'vnir')
    method = body.get('method', 'rf')
    n_clusters = int(body.get('n_clusters', 6))
    path = df.get_absolute_path()

    job = ProcessingJob.objects.create(
        data_file=df, user=request.user, job_type='classification',
        status='running', parameters=body)
    try:
        from .utils.analysis import (classify_random_forest, classify_svm,
                                      cluster_kmeans)
        data, _ = _read_cube_subsampled(df, path, sensor=sensor)

        # Generate training labels from k-means
        labels_map, _, _ = cluster_kmeans(data, n_clusters=n_clusters)

        if method == 'svm':
            class_map, metrics, info = classify_svm(data, labels_map)
        else:
            n_est = int(body.get('n_estimators', 100))
            class_map, metrics, info = classify_random_forest(
                data, labels_map, n_estimators=n_est)

        class_img = band_to_base64(class_map.astype(np.float32),
                                    colormap='tab10', title=f'{method.upper()} Classification')

        job.status = 'done'; job.log = info
        job.result_data = metrics
        job.completed_at = timezone.now(); job.save()

        return JsonResponse({
            'class_image': class_img, 'info': info,
            'metrics': metrics, 'n_classes': n_clusters,
        })
    except Exception as e:
        job.status = 'error'; job.log = traceback.format_exc(); job.save()
        return JsonResponse({'error': str(e)}, status=500)


@login_required
@csrf_exempt
def api_run_clustering(request, pk):
    """Run unsupervised clustering (K-means or Spectral)."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    df = get_object_or_404(DataFile, pk=pk, user=request.user)
    body = json.loads(request.body)
    sensor = body.get('sensor', 'vnir')
    method = body.get('method', 'kmeans')
    n_clusters = int(body.get('n_clusters', 8))
    path = df.get_absolute_path()

    job = ProcessingJob.objects.create(
        data_file=df, user=request.user, job_type='clustering',
        status='running', parameters=body)
    try:
        from .utils.analysis import cluster_kmeans, cluster_spectral
        data, _ = _read_cube_subsampled(df, path, sensor=sensor)

        if method == 'spectral':
            cluster_map, centroids, info = cluster_spectral(data, n_clusters)
        else:
            cluster_map, centroids, info = cluster_kmeans(data, n_clusters)

        cluster_img = band_to_base64(cluster_map.astype(np.float32),
                                      colormap='tab10',
                                      title=f'{method.title()} ({n_clusters} clusters)')

        # Centroid chart
        wls = _get_wavelengths(df, sensor) or list(range(data.shape[2]))
        c_list = [{'spectrum': c.tolist(), 'wavelengths': wls,
                    'label': f'C-{i+1}'} for i, c in enumerate(centroids)]
        centroid_chart = multi_profile_plotly_json(c_list)

        job.status = 'done'; job.log = info
        job.completed_at = timezone.now(); job.save()

        return JsonResponse({
            'cluster_image': cluster_img, 'centroid_chart': centroid_chart,
            'info': info, 'n_clusters': n_clusters,
        })
    except Exception as e:
        job.status = 'error'; job.log = traceback.format_exc(); job.save()
        return JsonResponse({'error': str(e)}, status=500)
