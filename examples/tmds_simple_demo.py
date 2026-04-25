import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.model_selection import train_test_split


def nearest_centroid_predict(X_train, y_train, X_test):
    centroids = np.array([X_train[y_train == k].mean(axis=0) for k in np.unique(y_train)])
    distances = ((X_test[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def gram_error_without_pairs(Tc, Z):
    # Appendix E: evaluate ||Tc Tc^T - Z Z^T||_F without forming an n x n matrix.
    sq_error = (
        np.linalg.norm(Tc.T @ Tc, "fro") ** 2
        + np.linalg.norm(Z.T @ Z, "fro") ** 2
        - 2 * np.linalg.norm(Tc.T @ Z, "fro") ** 2
    )
    return np.sqrt(max(sq_error, 0.0))


def gram_bounds(Tc, Z):
    E = Tc - Z
    eps = np.linalg.norm(E, "fro")
    tc_norm = np.linalg.norm(Tc, "fro")
    tc_norm_sq = tc_norm**2
    r2 = 1 - eps**2 / tc_norm_sq

    # Main text and Appendix E bounds, from loosest to tightest.
    frob_bound = 2 * tc_norm * eps + eps**2
    op_bound = 2 * np.linalg.norm(Tc, 2) * eps + eps**2
    cross_bound = 2 * np.linalg.norm(Tc.T @ E, "fro") + eps**2

    # Appendix E.3 normalized R^2 forms of the same three bounds.
    eta = eps / tc_norm
    alpha = np.linalg.norm(Tc, 2) / tc_norm
    beta = np.linalg.norm(Tc.T @ E, "fro") / (tc_norm * eps) if eps else 0.0
    r2_frob_bound = tc_norm_sq * (2 * eta + eta**2)
    r2_op_bound = tc_norm_sq * (2 * alpha * eta + eta**2)
    r2_cross_bound = tc_norm_sq * (2 * beta * eta + eta**2)

    return r2, cross_bound, op_bound, frob_bound, r2_cross_bound, r2_op_bound, r2_frob_bound


# Load iris dataset and prepare for training and testing.
iris = load_iris()
T = np.eye(3)[iris.target]

X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(
    iris.data, T, iris.target, test_size=0.30, stratify=iris.target, random_state=0
)

# Standardize from training statistics only, so held-out rows stay held out.
x_mean = X_train.mean(axis=0)
x_std = X_train.std(axis=0)
Y_train = (X_train - x_mean) / x_std
Y_test = (X_test - x_mean) / x_std

# TMDS works on centered features and centered targets; the train means define
# the intercept, and the same centering is reused out of sample.
y_mean = Y_train.mean(axis=0)
t_mean = T_train.mean(axis=0)
Yc_train = Y_train - y_mean
Yc_test = Y_test - y_mean
Tc_train = T_train - t_mean
Tc_test = T_test - t_mean

# Fit the multivariate ridge map B, then use it twice: prediction and geometry.
lam = 1e-6
B = np.linalg.solve(
    Yc_train.T @ Yc_train + lam * np.eye(Yc_train.shape[1]),
    Yc_train.T @ Tc_train,
)
Z_train = Yc_train @ B
Z_test = Yc_test @ B
M = B @ B.T

# Held-out clustering and nearest-centroid checks give a small practical signal.
raw_clusters = KMeans(n_clusters=3, n_init=20, random_state=0).fit_predict(Y_test)
tmds_clusters = KMeans(n_clusters=3, n_init=20, random_state=0).fit_predict(Z_test)
raw_centroid_pred = nearest_centroid_predict(Yc_train, y_train, Yc_test)
tmds_centroid_pred = nearest_centroid_predict(Z_train, y_train, Z_test)

metric_error = np.linalg.norm(Yc_test @ M @ Yc_test.T - Z_test @ Z_test.T, "fro")
gram_error = np.linalg.norm(Tc_test @ Tc_test.T - Z_test @ Z_test.T, "fro")
low_rank_gram_error = gram_error_without_pairs(Tc_test, Z_test)
(
    r2,
    cross_bound,
    op_bound,
    frob_bound,
    r2_cross_bound,
    r2_op_bound,
    r2_frob_bound,
) = gram_bounds(Tc_test, Z_test)

print(f"Train/test rows: {len(y_train)}/{len(y_test)}")
print(f"Held-out TMDS embedding shape: {Z_test.shape}")
print(f"Held-out target R^2: {r2:.3f}")
print(f"Held-out KMeans ARI, original features: {adjusted_rand_score(y_test, raw_clusters):.3f}")
print(f"Held-out KMeans ARI, TMDS space: {adjusted_rand_score(y_test, tmds_clusters):.3f}")
print(f"Held-out nearest-centroid accuracy, original features: {accuracy_score(y_test, raw_centroid_pred):.3f}")
print(f"Held-out nearest-centroid accuracy, TMDS space: {accuracy_score(y_test, tmds_centroid_pred):.3f}")
print(f"Learned metric M shape: {M.shape}")
print(f"Rank of learned metric M: {np.linalg.matrix_rank(M)}")
print(f"Held-out ||Yc M Yc^T - Z Z^T||_F: {metric_error:.2e}")
print(f"Held-out Gram error: {gram_error:.3f} ({low_rank_gram_error:.3f} without n x n)")
print(f"Appendix bounds: {gram_error:.3f} <= {cross_bound:.3f} <= {op_bound:.3f} <= {frob_bound:.3f}")
print(f"R^2 forms: cross {r2_cross_bound:.3f}, op {r2_op_bound:.3f}, frob {r2_frob_bound:.3f}")
