import numpy as np
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d


def alpha_clustering(features, alpha):
    centers = []
    labels = []

    for z in features:
        if len(centers) == 0:
            centers.append(z)
            labels.append(0)
            continue

        dists = [np.linalg.norm(z - c) for c in centers]
        idx = np.argmin(dists)

        if dists[idx] <= alpha:
            labels.append(idx)
            # mise à jour moyenne du cluster
            centers[idx] = (centers[idx] + z) / 2
        else:
            centers.append(z)
            labels.append(len(centers) - 1)

    return np.array(labels), np.array(centers)


class ABBA_like:
    def __init__(self, n_symbols=10, window_size=5):
        """
        n_symbols   : nombre de symboles de l'alphabet
        window_size : taille des segments pour approx. symbolique
        """
        self.n_symbols = n_symbols
        self.window_size = window_size
        self.kmeans = None
        self.symbol_map = None

    def fit(self, series):
        """
        series : np.array (1D)
        """
        # 1️⃣ Segmenter la série
        segments = []
        for i in range(len(series) - self.window_size + 1):
            seg = series[i:i+self.window_size]
            segments.append(seg)
        segments = np.array(segments)

        # 2️⃣ Dimension réduite : moyenne et pente par segment
        means = segments.mean(axis=1)
        slopes = (segments[:, -1] - segments[:, 0]) / self.window_size
        features = np.column_stack([means, slopes])

        # 3️⃣ Clustering pour créer l'alphabet
        self.kmeans = KMeans(n_clusters=self.n_symbols, random_state=42)
        self.kmeans.fit(features)
        labels = self.kmeans.labels_

        # 4️⃣ Mapping symbole -> segment moyen
        self.symbol_map = {}
        for sym in range(self.n_symbols):
            mask = labels == sym
            self.symbol_map[sym] = segments[mask].mean(axis=0)

        return labels

    def transform(self, series):
        """
        Transforme une nouvelle série en séquence symbolique
        """
        if self.kmeans is None:
            raise ValueError("ABBA non entraîné. Appeler fit() d'abord.")

        segments = []
        for i in range(len(series) - self.window_size + 1):
            seg = series[i:i+self.window_size]
            segments.append(seg)
        segments = np.array(segments)

        features = np.column_stack([
            segments.mean(axis=1),
            (segments[:, -1] - segments[:, 0]) / self.window_size
        ])

        labels = self.kmeans.predict(features)
        return labels

    def inverse_transform_smooth(self, symbols):
        window = self.window_size
        recon = np.zeros(len(symbols) + window - 1)
        counts = np.zeros_like(recon)

        for i, s in enumerate(symbols):
            segment = self.symbol_map[s]
            recon[i:i+window] += segment
            counts[i:i+window] += 1

        recon /= counts  # moyenne sur chevauchements
        return recon


class ABBA:
    def __init__(self, compression_tolerance=0.02, n_symbols=None):
        """
        compression_tolerance : tolérance ε pour fusion des segments
        n_symbols             : nombre de symboles de l'alphabet (optionnel)
        """
        self.epsilon = compression_tolerance
        self.n_symbols = n_symbols
        self.segments = []        # liste des segments (length, slope)
        self.symbol_map = {}      # symbole -> segment
        self.kmeans = None
        self.symbol_sequence = None

    # ----------------------------
    # 1️⃣ Segmentation adaptative
    # ----------------------------
    def _segment_series(self, series):
        segments = []
        start = 0
        n = len(series)
        
        while start < n - 1:
            # segment minimal
            end = start + 1
            slope = series[end] - series[start]
            
            # essayer d’étendre le segment tant que l’erreur ≤ ε
            while end + 1 < n:
                # prédiction linéaire
                seg_len = end - start + 1
                pred = series[start] + slope * np.arange(1, seg_len + 1) / seg_len
                err = np.max(np.abs(series[start:end+1] - pred))
                
                if err > self.epsilon:
                    break
                end += 1
                slope = series[end] - series[start]
            
            # enregistrer le segment : (start, end, slope)
            segments.append((start, end, slope))
            start = end
        
        return segments

    # ----------------------------
    # 2️⃣ Fit ABBA : segmentation + clustering
    # ----------------------------
    def fit(self, series):
        """
        series : np.array 1D
        Retourne : sequence symbolique
        """
        # 1️⃣ Segmentation adaptative
        segs = self._segment_series(series)
        
        # 2️⃣ Extraire features pour clustering : slope et length
        features = np.array([
            [seg[1]-seg[0], seg[2]]  # length, slope
            for seg in segs
        ])
        
        # 3️⃣ Clustering
        n_clusters = self.n_symbols or len(segs)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.kmeans.fit(features)
        labels = self.kmeans.labels_
        
        # 4️⃣ Construire mapping symbole -> segment moyen
        self.symbol_map = {}
        for sym in range(n_clusters):
            mask = labels == sym
            seg_means = features[mask]
            self.symbol_map[sym] = seg_means.mean(axis=0)  # length, slope
        
        # 5️⃣ Stocker la séquence symbolique
        self.symbol_sequence = labels
        return labels

    # ----------------------------
    # 3️⃣ Transform d’une nouvelle série
    # ----------------------------
    def transform(self, series):
        segs = self._segment_series(series)
        features = np.array([[seg[1]-seg[0], seg[2]] for seg in segs])
        return self.kmeans.predict(features)

    # ----------------------------
    # 4️⃣ Inverse transform : reconstruction
    # ----------------------------
    def inverse_transform(self, symbols, x0):
        series = []
        current = x0
        for s in symbols:
            s = int(s)   # conversion automatique
            length, slope = self.symbol_map[s]
            length = int(round(length))
            for _ in range(length):
                current = current + slope
                series.append(current)
        return np.array(series)


class ABBAPatched:
    """
    Implémentation fidèle de ABBA avec patched reconstruction
    (Elsworth & Güttel)
    """

    def __init__(self, compression_tol=0.01, alpha=0.1, max_k=None, random_state=42):
        """
        compression_tol  : tolérance de compression (epsilon dans l'article)
        """
        self.compression_tol = compression_tol
        self.random_state = random_state
        self.alpha = alpha
        self.max_k = max_k
        self.symbol_sequence = None
        self.patches = None
        self.cluster_centers = None

    # ==========================================================
    # 1️⃣ COMPRESSION (Adaptive piecewise linear approximation)
    # ==========================================================

    def _compress(self, series):
        """
        Retourne une liste de segments sous la forme :
        (i_start, i_end, length, increment)
        """
        segments = []
        i0 = 0
        N = len(series)

        while i0 < N - 1:
            i1 = i0 + 1
            while i1 < N:
                # droite candidate
                slope = (series[i1] - series[i0]) / (i1 - i0)
                approx = series[i0] + slope * np.arange(i1 - i0 + 1)
                error = np.max(np.abs(series[i0:i1+1] - approx))

                if error > self.compression_tol:
                    i1 -= 1
                    break

                i1 += 1

            if i1 <= i0:
                i1 = i0 + 1
            if i1 >= N:
                i1 -= 1
            
            length = i1 - i0
            inc = series[i1] - series[i0]

            segments.append((i0, i1, length, inc))
            i0 = i1

        return segments
    
    def _merge_clusters(self, labels, centers, features):
        # distance entre centres
        dists = np.linalg.norm(
            centers[:, None, :] - centers[None, :, :], axis=2
        )
        np.fill_diagonal(dists, np.inf)

        i, j = np.unravel_index(np.argmin(dists), dists.shape)

        # fusion j → i
        labels = labels.copy()
        labels[labels == j] = i
        labels[labels > j] -= 1
        unique = np.unique(labels)
        # recalcul centres
        new_centers = []
        for k in unique:
            idx = np.where(labels == k)[0]
            new_centers.append(features[idx].mean(axis=0))

        return labels, np.array(new_centers)

    # ==========================================================
    # 2️⃣ DIGITIZATION + PATCH CONSTRUCTION
    # ==========================================================
    def print_patches(self):
        for k, p in self.patches.items():
            print(f"Symbol {k}: length={len(p)}, delta={p[-1] - p[0]}")
        return None
        
    def fit(self, series):
        """
        series : np.array (1D)
        """
        series = np.asarray(series, dtype=float)
        segments = self._compress(series)
        self.lengths = [seg[2] for seg in segments]
        # features pour clustering
        features = np.array([[seg[2], seg[3]] for seg in segments])
        features = (features - features.mean(axis=0)) / features.std(axis=0)
        print("First two features:", features[:2])
        print("Distance:", np.linalg.norm(features[1] - features[0]))
        print("Alpha:", self.alpha)
        labels, centers = alpha_clustering(features, alpha=self.alpha)
        if self.max_k is not None:
            while len(centers) > self.max_k:
                print('nb of clusters :', len(centers))
                labels, centers = self._merge_clusters(
                    labels, centers, features
                )
        # unique = np.unique(labels)
        # mapping = {old: new for new, old in enumerate(unique)}
        # labels = np.array([mapping[l] for l in labels])
        self.cluster_centers = centers
        self.n_symbols = len(centers)

        self.symbol_sequence = labels

        unique_labels = np.unique(labels)
        self.patches = {}

        for sym in unique_labels:
            idx = np.where(labels == sym)[0]
            raw_segments = []
            lengths = []

            for i in idx:
                i0, i1, _, _ = segments[i]
                seg = series[i0:i1+1]
                raw_segments.append(seg)
                lengths.append(len(seg))

            L = int(round(np.mean(lengths)))

            aligned = []
            for seg in raw_segments:
                x_old = np.linspace(0, 1, len(seg))
                x_new = np.linspace(0, 1, L)
                f = interp1d(x_old, seg, kind="linear")
                aligned.append(f(x_new))
            self.patches[sym] = np.mean(aligned, axis=0)
        return labels

    # ==========================================================
    # 3️⃣ TRANSFORM (series → symbols)
    # ==========================================================

    def transform(self, series):
        series = np.asarray(series, dtype=float)
        segments = self._compress(series)
        features = np.array([[seg[2], seg[3]] for seg in segments])
        labels = np.array([np.argmin(np.linalg.norm(fc - self.cluster_centers, axis=1)) for fc in features])
        return labels

    # ==========================================================
    # 4️⃣ INVERSE TRANSFORM (PATCHED RECONSTRUCTION)
    # ==========================================================

    def inverse_transform(self, symbols, x0=None):
        """
        symbols : séquence symbolique
        x0      : valeur initiale (optionnelle)
        """
        recon = []
        current = x0

        for s in symbols:
            s = int(s)
            patch = self.patches[s].copy()

            if current is not None:
                patch = patch - patch[0] + current

            recon.extend(patch)
            current = recon[-1]

        return np.array(recon)