"""
Advanced analysis utilities for hyperspectral data.
PPI, SAM, LSU, SVM, Random Forest, K-means, Spectral Clustering.
"""
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDMEMBER EXTRACTION (PPI)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ppi(data, n_iterations=1000, threshold_percentile=99.5, sample_size=10000):
    """
    Pixel Purity Index (PPI) endmember extraction.
    Projects pixels onto random unit vectors and counts how often
    each pixel appears as an extremum.
    """
    rows, cols, bands = data.shape
    pixels = data.reshape(-1, bands).astype(np.float32)

    # Subsample for speed
    valid_mask = np.all(np.isfinite(pixels), axis=1) & np.any(pixels > 0, axis=1)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) > sample_size:
        sub_idx = np.random.choice(valid_idx, sample_size, replace=False)
    else:
        sub_idx = valid_idx

    sub_pixels = pixels[sub_idx]
    ppi_scores = np.zeros(len(sub_idx), dtype=np.int32)

    np.random.seed(42)
    for _ in range(n_iterations):
        vec = np.random.randn(bands).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-8
        projections = sub_pixels @ vec
        ppi_scores[np.argmax(projections)] += 1
        ppi_scores[np.argmin(projections)] += 1

    # Map scores back to full image
    full_scores = np.zeros(rows * cols, dtype=np.float32)
    full_scores[sub_idx] = ppi_scores.astype(np.float32)
    ppi_map = full_scores.reshape(rows, cols)

    # Extract endmembers
    threshold = np.percentile(ppi_scores[ppi_scores > 0], threshold_percentile) if np.any(ppi_scores > 0) else 1
    pure_mask = ppi_scores >= threshold
    pure_spectra = sub_pixels[pure_mask]

    # Cluster pure pixels into endmembers
    n_endmembers = min(10, max(2, len(pure_spectra) // 5))
    if len(pure_spectra) >= n_endmembers:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_endmembers, random_state=42, n_init=10)
        km.fit(pure_spectra)
        endmembers = km.cluster_centers_
    else:
        endmembers = pure_spectra[:max(1, len(pure_spectra))]

    info = (f"PPI: {n_iterations} iterations, {len(pure_spectra)} pure pixels found, "
            f"{len(endmembers)} endmembers extracted")
    return ppi_map, endmembers, info


# ═══════════════════════════════════════════════════════════════════════════════
#  TARGET DETECTION (SAM)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sam(data, target_spectrum):
    """
    Spectral Angle Mapper (SAM) target detection.
    Computes the spectral angle between each pixel and the target spectrum.
    Lower angle = more similar to target.
    """
    rows, cols, bands = data.shape
    pixels = data.reshape(-1, bands).astype(np.float64)
    target = np.array(target_spectrum, dtype=np.float64).flatten()

    # Compute spectral angles
    dot = pixels @ target
    norm_p = np.sqrt(np.sum(pixels ** 2, axis=1))
    norm_t = np.sqrt(np.sum(target ** 2))

    cos_angle = dot / (norm_p * norm_t + 1e-10)
    cos_angle = np.clip(cos_angle, -1, 1)
    angles = np.arccos(cos_angle)

    sam_map = angles.reshape(rows, cols)
    info = f"SAM: angle range [{np.nanmin(sam_map):.4f}, {np.nanmax(sam_map):.4f}] radians"
    return sam_map.astype(np.float32), info


def compute_sam_classification(data, endmembers, labels=None):
    """
    SAM-based classification — assigns each pixel to the closest endmember.
    Returns class map and angle maps for each endmember.
    """
    rows, cols, bands = data.shape
    n_em = len(endmembers)
    angle_maps = np.zeros((rows * cols, n_em), dtype=np.float32)

    pixels = data.reshape(-1, bands).astype(np.float64)
    for i, em in enumerate(endmembers):
        em = np.array(em, dtype=np.float64).flatten()
        dot = pixels @ em
        norm_p = np.sqrt(np.sum(pixels ** 2, axis=1))
        norm_t = np.sqrt(np.sum(em ** 2))
        cos_a = np.clip(dot / (norm_p * norm_t + 1e-10), -1, 1)
        angle_maps[:, i] = np.arccos(cos_a)

    class_map = np.argmin(angle_maps, axis=1).reshape(rows, cols)
    if labels is None:
        labels = [f"EM-{i+1}" for i in range(n_em)]

    info = f"SAM Classification: {n_em} endmembers, {rows*cols} pixels classified"
    return class_map.astype(np.int32), labels, info


# ═══════════════════════════════════════════════════════════════════════════════
#  SPECTRAL UNMIXING (LSU / NNLS)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_lsu(data, endmembers):
    """
    Linear Spectral Unmixing using Non-Negative Least Squares (NNLS).
    Estimates abundance fractions of each endmember per pixel.
    """
    from scipy.optimize import nnls
    rows, cols, bands = data.shape
    n_em = len(endmembers)
    em_matrix = np.array(endmembers, dtype=np.float64).T  # (bands, n_em)

    abundance = np.zeros((rows * cols, n_em), dtype=np.float32)
    pixels = data.reshape(-1, bands).astype(np.float64)

    # Process in chunks for memory efficiency
    chunk = 5000
    for start in range(0, len(pixels), chunk):
        end = min(start + chunk, len(pixels))
        for i in range(start, end):
            pixel = pixels[i]
            if np.any(pixel > 0) and np.all(np.isfinite(pixel)):
                try:
                    ab, _ = nnls(em_matrix, pixel)
                    total = ab.sum()
                    if total > 0:
                        ab /= total  # Normalize to sum=1
                    abundance[i] = ab.astype(np.float32)
                except Exception:
                    pass

    abundance_cube = abundance.reshape(rows, cols, n_em)
    info = (f"LSU/NNLS: {n_em} endmembers, abundance range "
            f"[{abundance.min():.4f}, {abundance.max():.4f}]")
    return abundance_cube, info


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION (SVM, Random Forest)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_random_forest(data, labels_map, n_estimators=100, test_size=0.3):
    """
    Random Forest classification on hyperspectral data.
    Uses dominant material map (from PPI+SAM or clustering) as labels.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    rows, cols, bands = data.shape
    X = data.reshape(-1, bands).astype(np.float32)
    y = labels_map.flatten().astype(np.int32)

    # Filter valid pixels
    valid = np.all(np.isfinite(X), axis=1) & (y >= 0)
    X_v, y_v = X[valid], y[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X_v, y_v, test_size=test_size, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=20,
                                  random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    # Predict full image
    pred = np.full(len(X), -1, dtype=np.int32)
    pred[valid] = clf.predict(X_v)
    class_map = pred.reshape(rows, cols)

    info = (f"RF: {n_estimators} trees, train_acc={train_acc:.4f}, "
            f"test_acc={test_acc:.4f}, {len(np.unique(y_v))} classes")
    return class_map, {'train_acc': train_acc, 'test_acc': test_acc}, info


def classify_svm(data, labels_map, sample_size=20000):
    """
    SVM classification on hyperspectral data.
    Subsamples for training due to SVM's O(n²) complexity.
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    rows, cols, bands = data.shape
    X = data.reshape(-1, bands).astype(np.float32)
    y = labels_map.flatten().astype(np.int32)

    valid = np.all(np.isfinite(X), axis=1) & (y >= 0)
    X_v, y_v = X[valid], y[valid]

    # Subsample for SVM
    if len(X_v) > sample_size:
        idx = np.random.choice(len(X_v), sample_size, replace=False)
        X_s, y_s = X_v[idx], y_v[idx]
    else:
        X_s, y_s = X_v, y_v

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_s)

    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y_s, test_size=0.3, random_state=42
    )

    clf = SVC(kernel='rbf', gamma='scale', random_state=42)
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    # Predict full image
    X_scaled = scaler.transform(X_v)
    pred = np.full(len(X), -1, dtype=np.int32)
    pred[valid] = clf.predict(X_scaled)
    class_map = pred.reshape(rows, cols)

    info = (f"SVM (RBF): train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, "
            f"sampled {len(X_s)} pixels")
    return class_map, {'train_acc': train_acc, 'test_acc': test_acc}, info


# ═══════════════════════════════════════════════════════════════════════════════
#  CLUSTERING (K-Means, Spectral)
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_kmeans(data, n_clusters=8, sample_size=50000):
    """K-means clustering on hyperspectral cube."""
    from sklearn.cluster import MiniBatchKMeans

    rows, cols, bands = data.shape
    X = data.reshape(-1, bands).astype(np.float32)
    valid = np.all(np.isfinite(X), axis=1)
    X_v = X[valid]

    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                          batch_size=min(10000, len(X_v)), n_init=3)
    km.fit(X_v)
    labels = np.full(len(X), -1, dtype=np.int32)
    labels[valid] = km.labels_
    cluster_map = labels.reshape(rows, cols)

    info = (f"K-means: {n_clusters} clusters, inertia={km.inertia_:.2f}, "
            f"{len(X_v)} valid pixels")
    return cluster_map, km.cluster_centers_, info


def cluster_spectral(data, n_clusters=6, sample_size=5000):
    """
    Spectral Clustering on a subsample of pixels.
    Projects result back to full image using nearest-neighbor assignment.
    """
    from sklearn.cluster import SpectralClustering
    from sklearn.neighbors import NearestCentroid

    rows, cols, bands = data.shape
    X = data.reshape(-1, bands).astype(np.float32)
    valid = np.all(np.isfinite(X), axis=1) & np.any(X > 0, axis=1)
    X_v = X[valid]

    # Subsample (spectral clustering is O(n³))
    if len(X_v) > sample_size:
        idx = np.random.choice(len(X_v), sample_size, replace=False)
        X_s = X_v[idx]
    else:
        X_s = X_v
        idx = np.arange(len(X_v))

    sc = SpectralClustering(n_clusters=n_clusters, random_state=42,
                             affinity='nearest_neighbors', n_neighbors=10,
                             assign_labels='kmeans')
    sub_labels = sc.fit_predict(X_s)

    # Compute centroids and assign all pixels
    centroids = np.zeros((n_clusters, bands), dtype=np.float32)
    for k in range(n_clusters):
        mask = sub_labels == k
        if np.any(mask):
            centroids[k] = X_s[mask].mean(axis=0)

    # Assign all valid pixels to nearest centroid
    dists = np.linalg.norm(X_v[:, None, :] - centroids[None, :, :], axis=2)
    full_labels = np.full(len(X), -1, dtype=np.int32)
    full_labels[valid] = np.argmin(dists, axis=1)
    cluster_map = full_labels.reshape(rows, cols)

    info = (f"Spectral Clustering: {n_clusters} clusters, "
            f"sampled {len(X_s)} of {len(X_v)} pixels")
    return cluster_map, centroids, info
