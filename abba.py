import numpy as np
from sklearn.cluster import KMeans


class ABBA:
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
