"""
Visualization utilities: band images, RGB composites, spectral plots, histograms.
All return base64-encoded PNG strings or Plotly JSON for Django views.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import base64
import json


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _fig_to_base64(fig, dpi=100, tight=True, bg='#0a0e1a'):
    """Convert matplotlib figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.patch.set_facecolor(bg)
    if tight:
        plt.tight_layout()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=bg, edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _stretch_band(band, p_low=2, p_high=98):
    """Percentile stretch a 2D band to [0,1]."""
    band = band.astype(np.float32)
    valid = band[np.isfinite(band)]
    if len(valid) == 0:
        return np.zeros_like(band)
    lo = np.percentile(valid, p_low)
    hi = np.percentile(valid, p_high)
    if hi <= lo:
        return np.zeros_like(band)
    stretched = np.clip((band - lo) / (hi - lo), 0, 1)
    return stretched


# ─── Single Band Image ────────────────────────────────────────────────────────

def band_to_base64(band_data, colormap='inferno', title='Band Image', wavelength=None):
    """
    Render a single 2D band as a colorized PNG.
    Returns base64 string.
    """
    stretched = _stretch_band(band_data)
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0a0e1a')
    ax.set_facecolor('#0a0e1a')
    im = ax.imshow(stretched, cmap=colormap, interpolation='nearest', aspect='auto')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='#a0b4c8')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#a0b4c8', fontsize=8)
    cbar.set_label('Normalized DN', color='#a0b4c8', fontsize=9)
    wl_str = f" | λ={wavelength:.1f}nm" if wavelength else ""
    ax.set_title(f"{title}{wl_str}", color='#e0e8f0', fontsize=11, pad=8)
    ax.tick_params(colors='#a0b4c8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5a')
    plt.tight_layout(pad=0.5)
    return _fig_to_base64(fig, dpi=100)


# ─── RGB False-Color Composite ────────────────────────────────────────────────

def rgb_composite_to_base64(data, r_idx, g_idx, b_idx, title='RGB Composite'):
    """
    Create false-color RGB composite from three bands.
    data: (rows, cols, bands) float array
    Returns base64 PNG.
    """
    r = _stretch_band(data[:, :, r_idx])
    g = _stretch_band(data[:, :, g_idx])
    b_ch = _stretch_band(data[:, :, b_idx])
    rgb = np.stack([r, g, b_ch], axis=-1)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0a0e1a')
    ax.set_facecolor('#0a0e1a')
    ax.imshow(rgb, interpolation='nearest', aspect='auto')
    ax.set_title(title, color='#e0e8f0', fontsize=11, pad=8)
    ax.tick_params(colors='#a0b4c8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5a')
    return _fig_to_base64(fig, dpi=100)


# ─── Spectral Profile Plot ────────────────────────────────────────────────────

def spectral_profile_plotly_json(spectrum, wavelengths, label='Pixel Spectrum',
                                  row=None, col=None, color='#00d4ff'):
    """
    Return Plotly chart JSON for a spectral profile.
    Compatible with Plotly.js.
    """
    wl = wavelengths if wavelengths else list(range(len(spectrum)))
    title = f"Spectral Profile"
    if row is not None and col is not None:
        title += f" at Pixel ({row}, {col})"

    trace = {
        'x': [float(w) for w in wl],
        'y': [float(v) for v in spectrum],
        'mode': 'lines+markers',
        'name': label,
        'line': {'color': color, 'width': 2},
        'marker': {'size': 4, 'color': color},
        'hovertemplate': 'λ=%{x:.1f}nm<br>Value=%{y:.4f}<extra></extra>',
    }

    layout = {
        'title': {'text': title, 'font': {'color': '#e0e8f0', 'size': 14}},
        'paper_bgcolor': '#0d1926',
        'plot_bgcolor': '#111d2e',
        'xaxis': {
            'title': 'Wavelength (nm)',
            'color': '#a0b4c8',
            'gridcolor': '#1e3a5a',
            'zerolinecolor': '#1e3a5a',
        },
        'yaxis': {
            'title': 'Radiance / Reflectance',
            'color': '#a0b4c8',
            'gridcolor': '#1e3a5a',
            'zerolinecolor': '#1e3a5a',
        },
        'font': {'color': '#a0b4c8'},
        'margin': {'l': 60, 'r': 20, 't': 50, 'b': 60},
        'hovermode': 'x unified',
    }

    return json.dumps({'data': [trace], 'layout': layout})


def multi_profile_plotly_json(profiles):
    """
    Overlay multiple spectral profiles on one chart.
    profiles: list of dicts with keys: spectrum, wavelengths, label, color
    """
    colors = ['#00d4ff', '#39ff14', '#ff6b35', '#b44fff', '#ffeb3b',
              '#ff4081', '#00e5ff', '#76ff03']
    traces = []
    for i, prof in enumerate(profiles):
        color = prof.get('color', colors[i % len(colors)])
        traces.append({
            'x': [float(w) for w in prof['wavelengths']],
            'y': [float(v) for v in prof['spectrum']],
            'mode': 'lines',
            'name': prof.get('label', f'Profile {i+1}'),
            'line': {'color': color, 'width': 2},
            'hovertemplate': 'λ=%{x:.1f}nm<br>%{y:.4f}<extra></extra>',
        })

    layout = {
        'title': {'text': 'Spectral Profiles Comparison', 'font': {'color': '#e0e8f0', 'size': 14}},
        'paper_bgcolor': '#0d1926',
        'plot_bgcolor': '#111d2e',
        'xaxis': {
            'title': 'Wavelength (nm)',
            'color': '#a0b4c8',
            'gridcolor': '#1e3a5a',
        },
        'yaxis': {
            'title': 'Value',
            'color': '#a0b4c8',
            'gridcolor': '#1e3a5a',
        },
        'font': {'color': '#a0b4c8'},
        'margin': {'l': 60, 'r': 20, 't': 50, 'b': 60},
        'legend': {'bgcolor': '#0d1926', 'bordercolor': '#1e3a5a'},
        'hovermode': 'x unified',
    }

    return json.dumps({'data': traces, 'layout': layout})


# ─── Index Map ────────────────────────────────────────────────────────────────

def index_map_to_base64(index_data, title='Spectral Index', colormap='RdYlGn', vmin=-1, vmax=1):
    """
    Render a spectral index map as a colorized PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0a0e1a')
    ax.set_facecolor('#0a0e1a')
    im = ax.imshow(index_data, cmap=colormap, vmin=vmin, vmax=vmax,
                   interpolation='nearest', aspect='auto')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='#a0b4c8')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#a0b4c8', fontsize=8)
    cbar.set_label(title, color='#a0b4c8', fontsize=9)
    ax.set_title(title, color='#e0e8f0', fontsize=11, pad=8)
    ax.tick_params(colors='#a0b4c8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5a')
    return _fig_to_base64(fig, dpi=100)


# ─── Histogram ────────────────────────────────────────────────────────────────

def band_histogram_plotly_json(band_data, title='Band Histogram', n_bins=100, color='#00d4ff'):
    """Return Plotly histogram JSON for a single band."""
    valid = band_data.flatten()
    valid = valid[np.isfinite(valid)]

    # Compute histogram
    counts, edges = np.histogram(valid, bins=n_bins)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()

    trace = {
        'x': [float(c) for c in centers],
        'y': [int(c) for c in counts],
        'type': 'bar',
        'name': 'Frequency',
        'marker': {'color': color, 'opacity': 0.8},
        'hovertemplate': 'Value=%{x:.4f}<br>Count=%{y}<extra></extra>',
    }

    layout = {
        'title': {'text': title, 'font': {'color': '#e0e8f0', 'size': 13}},
        'paper_bgcolor': '#0d1926',
        'plot_bgcolor': '#111d2e',
        'xaxis': {'title': 'Value', 'color': '#a0b4c8', 'gridcolor': '#1e3a5a'},
        'yaxis': {'title': 'Count', 'color': '#a0b4c8', 'gridcolor': '#1e3a5a'},
        'font': {'color': '#a0b4c8'},
        'margin': {'l': 60, 'r': 20, 't': 50, 'b': 60},
        'bargap': 0.05,
    }

    return json.dumps({'data': [trace], 'layout': layout})


def band_stats_chart_plotly_json(stats, wavelengths=None):
    """Plot mean ± std across all bands as a Plotly line chart."""
    bands = [s['band'] for s in stats]
    means = [s['mean'] for s in stats]
    stds = [s['std'] for s in stats]
    x_vals = [wavelengths[b] if wavelengths and b < len(wavelengths) else b for b in bands]

    traces = [
        {
            'x': x_vals,
            'y': means,
            'mode': 'lines',
            'name': 'Mean',
            'line': {'color': '#00d4ff', 'width': 2},
        },
        {
            'x': x_vals + x_vals[::-1],
            'y': [m + s for m, s in zip(means, stds)] + [m - s for m, s in zip(means, stds)][::-1],
            'fill': 'toself',
            'fillcolor': 'rgba(0,212,255,0.15)',
            'line': {'color': 'rgba(0,0,0,0)'},
            'name': '±1 Std Dev',
            'showlegend': True,
        }
    ]

    layout = {
        'title': {'text': 'Band-wise Mean ± Std Dev', 'font': {'color': '#e0e8f0', 'size': 13}},
        'paper_bgcolor': '#0d1926',
        'plot_bgcolor': '#111d2e',
        'xaxis': {
            'title': 'Wavelength (nm)' if wavelengths else 'Band Index',
            'color': '#a0b4c8',
            'gridcolor': '#1e3a5a',
        },
        'yaxis': {'title': 'Radiance', 'color': '#a0b4c8', 'gridcolor': '#1e3a5a'},
        'font': {'color': '#a0b4c8'},
        'margin': {'l': 60, 'r': 20, 't': 50, 'b': 60},
        'legend': {'bgcolor': '#0d1926'},
    }

    return json.dumps({'data': traces, 'layout': layout})


# ─── PCA Visualization ────────────────────────────────────────────────────────

def pca_variance_plotly_json(explained_variance):
    """Plot PCA explained variance ratio."""
    n = len(explained_variance)
    cumulative = np.cumsum(explained_variance).tolist()

    traces = [
        {
            'x': list(range(1, n+1)),
            'y': [v*100 for v in explained_variance],
            'type': 'bar',
            'name': 'Individual',
            'marker': {'color': '#00d4ff'},
        },
        {
            'x': list(range(1, n+1)),
            'y': [v*100 for v in cumulative],
            'mode': 'lines+markers',
            'name': 'Cumulative',
            'line': {'color': '#39ff14', 'width': 2},
            'marker': {'size': 6},
            'yaxis': 'y2',
        }
    ]

    layout = {
        'title': {'text': 'PCA Explained Variance', 'font': {'color': '#e0e8f0', 'size': 13}},
        'paper_bgcolor': '#0d1926',
        'plot_bgcolor': '#111d2e',
        'xaxis': {'title': 'Principal Component', 'color': '#a0b4c8', 'gridcolor': '#1e3a5a'},
        'yaxis': {'title': 'Variance (%)', 'color': '#a0b4c8', 'gridcolor': '#1e3a5a'},
        'yaxis2': {
            'title': 'Cumulative (%)',
            'overlaying': 'y',
            'side': 'right',
            'color': '#39ff14',
            'gridcolor': 'rgba(0,0,0,0)',
        },
        'font': {'color': '#a0b4c8'},
        'margin': {'l': 60, 'r': 60, 't': 50, 'b': 60},
        'legend': {'bgcolor': '#0d1926'},
        'barmode': 'group',
    }

    return json.dumps({'data': traces, 'layout': layout})
