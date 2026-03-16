from sklearn.base import TransformerMixin
import numpy as np
from scipy.stats import multivariate_normal

class GmmMml(TransformerMixin):
    def __init__(self, kmin=1, kmax=25, regularize=1e-6, threshold=1e-5, covoption=0, max_iters=100, live_2d_plot=False, plots=False, variance_threshold=1e-8):
        self.kmin = kmin
        self.kmax = kmax
        self.regularize = regularize
        self.th = threshold
        self.covoption = covoption
        self.live_2d_plot = live_2d_plot
        self.max_iters = max_iters
        self.check_plot = plots
        self.variance_threshold = variance_threshold

    def _posterior_probability(self, y, estmu, estcov, i):
        try:
            return multivariate_normal.pdf(y, estmu[i], estcov[:, :, i], allow_singular=True)
        except Exception:
            return np.zeros(y.shape[0])

    def fit(self, X, y=None, verb=False):
        X = np.asarray(X)
        # Filter empty histograms (rows with all zeros)
        non_empty_mask = np.any(X != 0, axis=1)
        X = X[non_empty_mask]
        if X.shape[0] == 0:
            print("All input histograms are empty, skipping fit.")
            self.fitted = False
            return self

        variances = np.var(X, axis=0)
        significant_features = np.where(variances > self.variance_threshold)[0]
        insignificant_features = np.where(variances <= self.variance_threshold)[0]

        print(f"Significant features indices: {significant_features.tolist()}")
        print(f"Insignificant features indices: {insignificant_features.tolist()}")

        X = X[:, significant_features]
        npoints, dimens = X.shape
        k = self.kmax
        estmu = X[np.random.choice(npoints, k, replace=False)]
        estpp = np.full((1, k), 1.0 / k)
        globcov = np.cov(X, rowvar=False) + self.regularize * np.eye(dimens)
        estcov = np.stack([globcov for _ in range(k)], axis=2)

        for iteration in range(self.max_iters):
            semi_indic = np.array([self._posterior_probability(X, estmu, estcov, i) for i in range(k)])
            indic = semi_indic * estpp
            normindic = indic / (np.sum(indic, axis=0, keepdims=True) + np.finfo(float).eps)

            for i in range(k):
                weight = np.sum(normindic[i, :])
                if weight < 1e-8:
                    continue
                estmu[i] = np.sum(normindic[i, :, np.newaxis] * X, axis=0) / weight
                diff = X - estmu[i]
                estcov[:, :, i] = (normindic[i, :, np.newaxis] * diff).T @ diff / weight + self.regularize * np.eye(dimens)
                estpp[0, i] = weight / npoints

            estpp /= np.sum(estpp)
            if verb:
                print(f"Iteration {iteration}: estpp={estpp}")

        self.bestmu = estmu
        self.bestcov = estcov
        self.bestpp = estpp
        self.bestk = k
        self.significant_features = significant_features
        self.fitted = True
        return self

    def transform(self, X, y=None):
        if not hasattr(self, 'fitted') or not self.fitted:
            print("Model was not fitted due to empty or invalid input.")
            return np.zeros((X.shape[0], 1))
        X = np.asarray(X)
        X = X[:, self.significant_features]
        semi_indic = np.array([self._posterior_probability(X, self.bestmu, self.bestcov, i) for i in range(self.bestk)])
        return semi_indic.T

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def predict_proba(self, X):
        return self.transform(X)

    def predict(self, X):
        return np.argmax(self.transform(X), axis=1)
