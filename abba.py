import numpy as np
from sklearn.cluster import KMeans


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