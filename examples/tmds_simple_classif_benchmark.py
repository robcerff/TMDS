import argparse
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.model_selection import train_test_split


def print_table(title, headers, rows, right_align=()):
    right_align = set(right_align)
    widths = [
        max(len(str(value)) for value in column)
        for column in zip(headers, *rows)
    ]
    header = "  ".join(str(value).ljust(width) for value, width in zip(headers, widths))
    rule = "  ".join("-" * width for width in widths)

    print(f"\n{title}")
    print(header)
    print(rule)
    for row in rows:
        cells = []
        for header_name, value, width in zip(headers, row, widths):
            text = str(value)
            cells.append(text.rjust(width) if header_name in right_align else text.ljust(width))
        print("  ".join(cells))


def nearest_centroid_predict(X_train, y_train, X_test):
    classes = np.unique(y_train)
    centroids = np.array([X_train[y_train == c].mean(axis=0) for c in classes])
    sq_dist = ((X_test[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return classes[np.argmin(sq_dist, axis=1)]


def _fmt_metric(value, width=10):
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def _array_bytes(*arrays):
    return sum(array.nbytes for array in arrays)


def _array_mib(*arrays):
    return _array_bytes(*arrays) / 1024**2


def _lda_model_mib(lda):
    attrs = ("coef_", "intercept_", "means_", "scalings_", "xbar_")
    return sum(getattr(lda, attr).nbytes for attr in attrs if hasattr(lda, attr)) / 1024**2


def load_fashion_mnist(max_rows):
    try:
        X, y = fetch_openml(
            "Fashion-MNIST",
            version=1,
            return_X_y=True,
            as_frame=False,
            parser="liac-arff",
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load Fashion-MNIST from scikit-learn/OpenML. "
            "Check network access on first run, or retry after OpenML is reachable."
        ) from exc

    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))
    X = np.asarray(X[order], dtype=np.float64)
    y = np.asarray(y[order])
    if max_rows:
        X, y = X[:max_rows], y[:max_rows]
    return X, y


def gram_error_without_pairs(Tc, Z):
    # Appendix E: compute ||Tc Tc^T - Z Z^T||_F from q x q cross-products.
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

    cross_bound = 2 * np.linalg.norm(Tc.T @ E, "fro") + eps**2
    op_bound = 2 * np.linalg.norm(Tc, 2) * eps + eps**2
    frob_bound = 2 * tc_norm * eps + eps**2
    return r2, cross_bound, op_bound, frob_bound


def main():
    total_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="TMDS classification benchmark on Fashion-MNIST.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap after loading.")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--lambda", dest="lam", type=float, default=1e-6)
    parser.add_argument("--probe-rows", type=int, default=2000, help="Rows used for the small metric identity probe.")
    args = parser.parse_args()

    start = time.perf_counter()
    X, y = load_fashion_mnist(args.max_rows)
    load_seconds = time.perf_counter() - start

    _, y = np.unique(y, return_inverse=True)
    T = np.eye(len(np.unique(y)))[y]
    X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(
        X, T, y, test_size=args.test_size, stratify=y, random_state=0
    )

    fit_start = time.perf_counter()

    # Standardize and center using training statistics only; the same offsets are
    # reused for held-out rows, so the learned geometry is genuinely out of sample.
    x_mean = X_train.mean(axis=0)
    x_std = np.where(X_train.std(axis=0) == 0, 1, X_train.std(axis=0))
    Y_train = (X_train - x_mean) / x_std
    Y_test = (X_test - x_mean) / x_std

    y_mean = Y_train.mean(axis=0)
    t_mean = T_train.mean(axis=0)
    Yc_train = Y_train - y_mean
    Yc_test = Y_test - y_mean
    Tc_train = T_train - t_mean
    Tc_test = T_test - t_mean
    raw_centroid_pred = nearest_centroid_predict(Y_train, y_train, Y_test)

    # One p x p ridge solve learns B. No n x n distance or Gram matrix is formed.
    B = np.linalg.solve(
        Yc_train.T @ Yc_train + args.lam * np.eye(Yc_train.shape[1]),
        Yc_train.T @ Tc_train,
    )
    Z_test = Yc_test @ B
    M = B @ B.T
    tmds_class_pred = (t_mean + Z_test).argmax(axis=1)
    fit_seconds = time.perf_counter() - fit_start

    lda_start = time.perf_counter()
    lda = LinearDiscriminantAnalysis()
    lda.fit(Y_train, y_train)
    L_test = lda.transform(Y_test)
    lda_class_pred = lda.predict(Y_test)
    lda_seconds = time.perf_counter() - lda_start

    score_start = time.perf_counter()
    raw_clusters = MiniBatchKMeans(
        n_clusters=T.shape[1], n_init=5, random_state=0
    ).fit_predict(Y_test)
    raw_cluster_seconds = time.perf_counter() - score_start
    cluster_start = time.perf_counter()
    tmds_clusters = MiniBatchKMeans(
        n_clusters=T.shape[1], n_init=5, random_state=0
    ).fit_predict(Z_test)
    tmds_cluster_seconds = time.perf_counter() - cluster_start
    cluster_start = time.perf_counter()
    lda_clusters = MiniBatchKMeans(
        n_clusters=T.shape[1], n_init=5, random_state=0
    ).fit_predict(L_test)
    lda_cluster_seconds = time.perf_counter() - cluster_start
    kmeans_seconds = raw_cluster_seconds + tmds_cluster_seconds + lda_cluster_seconds
    majority_class = np.bincount(y_train).argmax()

    probe_n = min(args.probe_rows, len(y_test))
    metric_error = np.linalg.norm(
        Yc_test[:probe_n] @ M @ Yc_test[:probe_n].T - Z_test[:probe_n] @ Z_test[:probe_n].T,
        "fro",
    )
    gram_error = gram_error_without_pairs(Tc_test, Z_test)
    r2, cross_bound, op_bound, frob_bound = gram_bounds(Tc_test, Z_test)

    n, p = X.shape
    q = T.shape[1]
    pairwise_gib = n * n * 8 / 1024**3
    model_mib = (B.nbytes + M.nbytes) / 1024**2
    shared_mib = _array_mib(
        X,
        y,
        T,
        X_train,
        X_test,
        T_train,
        T_test,
        y_train,
        y_test,
        Y_train,
        Y_test,
        Yc_train,
        Yc_test,
        Tc_train,
        Tc_test,
        x_mean,
        x_std,
        y_mean,
        t_mean,
        raw_centroid_pred,
        raw_clusters,
        majority_class,
    )
    tmds_specific_mib = _array_mib(
        Z_test,
        B,
        M,
        tmds_class_pred,
        tmds_clusters,
    )
    lda_specific_mib = _array_mib(L_test, lda_class_pred, lda_clusters) + _lda_model_mib(lda)
    dense_arrays_mib = shared_mib + tmds_specific_mib + lda_specific_mib

    baseline_acc = accuracy_score(y_test, np.full_like(y_test, majority_class))
    raw_centroid_acc = accuracy_score(y_test, raw_centroid_pred)
    tmds_acc = accuracy_score(y_test, tmds_class_pred)
    lda_acc = accuracy_score(y_test, lda_class_pred)
    raw_ari = adjusted_rand_score(y_test, raw_clusters)
    tmds_ari = adjusted_rand_score(y_test, tmds_clusters)
    lda_ari = adjusted_rand_score(y_test, lda_clusters)

    print_table(
        "Data",
        ("Item", "Value"),
        (
            ("Rows/features/classes", f"{n:,}/{p}/{q}"),
            ("Dataset", "Fashion-MNIST"),
            ("Train/test rows", f"{len(y_train):,}/{len(y_test):,}"),
            ("TMDS embedding shape", str(Z_test.shape)),
            ("LDA embedding shape", str(L_test.shape)),
        ),
        right_align=("Value",),
    )
    print_table(
        "Memory",
        ("Item", "Value"),
        (
            ("Explicit n x n float64 matrix", f"{pairwise_gib:,.1f} GiB"),
            ("Dense arrays represented", f"{dense_arrays_mib:,.1f} MiB"),
            ("Shared data/preprocessing arrays", f"{shared_mib:,.1f} MiB"),
            ("TMDS-specific arrays", f"{tmds_specific_mib:,.1f} MiB"),
            ("LDA-specific arrays", f"{lda_specific_mib:,.1f} MiB"),
            ("TMDS learned B and M storage", f"{model_mib:,.3f} MiB"),
        ),
        right_align=("Value",),
    )
    print_table(
        "Wall-clock Seconds",
        ("Step", "Seconds"),
        (
            ("Load data", f"{load_seconds:.1f}"),
            ("TMDS fit/transform", f"{fit_seconds:.1f}"),
            ("LDA fit/transform", f"{lda_seconds:.1f}"),
            ("Original KMeans", f"{raw_cluster_seconds:.1f}"),
            ("TMDS KMeans", f"{tmds_cluster_seconds:.1f}"),
            ("LDA KMeans", f"{lda_cluster_seconds:.1f}"),
            ("KMeans total", f"{kmeans_seconds:.1f}"),
            ("Script total", f"{time.perf_counter() - total_start:.1f}"),
        ),
        right_align=("Seconds",),
    )
    print_table(
        "Held-out Comparison (higher is better)",
        ("Method", "Accuracy", "ARI", "Notes"),
        (
            ("Baseline majority class", _fmt_metric(baseline_acc), _fmt_metric(None), "No embedding"),
            ("Original nearest-centroid", _fmt_metric(raw_centroid_acc), _fmt_metric(None), "No embedding"),
            ("TMDS ridge-map classifier", _fmt_metric(tmds_acc), _fmt_metric(None), "Argmax on t_mean + Z_test"),
            ("LDA classifier", _fmt_metric(lda_acc), _fmt_metric(None), "LinearDiscriminantAnalysis predict"),
            ("Original MiniBatchKMeans", _fmt_metric(None), _fmt_metric(raw_ari), "Raw standardized features"),
            ("TMDS MiniBatchKMeans", _fmt_metric(None), _fmt_metric(tmds_ari), "TMDS embedding"),
            ("LDA MiniBatchKMeans", _fmt_metric(None), _fmt_metric(lda_ari), "LDA embedding"),
        ),
        right_align=("Accuracy", "ARI"),
    )
    print_table(
        "TMDS Diagnostics",
        ("Item", "Value"),
        (
            ("Held-out target R^2", f"{r2:.3f}"),
            ("Learned metric M shape/rank", f"{M.shape}/{np.linalg.matrix_rank(M)}"),
            (f"Probe ||Yc M Yc^T - Z Z^T||_F on {probe_n:,} rows", f"{metric_error:.2e}"),
            ("Held-out Gram error without n x n", f"{gram_error:,.1f}"),
            ("Appendix bounds", f"{gram_error:,.1f} <= {cross_bound:,.1f} <= {op_bound:,.1f} <= {frob_bound:,.1f}"),
        ),
    )


if __name__ == "__main__":
    main()
