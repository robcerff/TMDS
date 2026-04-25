# TMDS: Target Multidimensional Scaling

**No Distance Matrix Required: Multivariate Regression as Supervised Metric Learning**

Robert Cerff · April 2026

---

TMDS is a supervised metric learning framework that produces target-aligned clusters and out-of-sample embeddings without forming an n × n pairwise distance matrix or Gram matrix. It learns a Mahalanobis-style metric from a multivariate ridge regression map, giving a scalable alternative to supervised MDS when quadratic memory is the bottleneck.

We built TMDS because target-aligned clustering becomes expensive at scale. Methods that tie cluster structure to supervised outcomes often rely on pairwise distances or Gram matrices, so cost grows quadratically with n. At a million observations that's already uncomfortable; at ten million it's impractical. TMDS avoids this entirely: the fit is a single linear solve, the clusters are mathematically consistent with the predictions, and the cost stays linear in n.

What's surprising is that the same object does both jobs at once: the ridge map B̂ is simultaneously a multivariate predictor and the factor of a PSD metric (M = B̂ B̂ᵀ). Supervision and structure fall out of the same fit, and since the scaling argument rests on how the fit is organized rather than on linearity itself, it extends cleanly to non-linear regimes.

The framework lives in the squared-Euclidean geometry underlying MDS-style methods. That still covers learned metrics, embeddings, clustering, prediction, non-linear bases, and rank-truncated representations, while keeping the theory tied to PSD metrics and Gram/Frobenius error.

## Intuition

The target defines what similarity should mean. Without a target, similarity defaults to Euclidean proximity in the original feature space. With a target, we are saying something more specific: during training we observe both features and outcomes, but out of sample we only observe features yet still care about outcome behavior.

TMDS uses training data to align feature geometry with target geometry, so rows that are close under the learned metric are also close in target space — even when the target is no longer observed.

In practice this gives:

- **Scaling**
  - **No quadratic object:** the fit works with feature-by-feature and feature-by-target matrices, avoiding explicit n × n distance or Gram matrices.
  - **CPU/GPU friendly:** the core operations are dense matrix multiplies and linear solves, which run naturally on CPU and can move to GPU-backed array libraries.

---

- **Training**
  - **Mixed target types:** categorical labels, continuous responses, and multivariate outcomes can all be represented in the target matrix.
  - **Non-linear extensions:** explicit input bases can come from feature libraries, random features, neural representations, or other representation learning models, providing a non-linear version of TMDS with the same favorable scaling in n.
  - **Out-of-sample by construction:** new unseen rows use the same learned map instead of requiring a new fit.

---

- **Diagnostics**
  - **Error guarantees:** R^2 fits deterministic bounds on centered Gram error, so prediction fit directly controls target-geometry fidelity.
  - **Rank diagnostics:** rank truncation gives a compact supervised representation; when the discarded singular-value tail is small, the fitted target geometry is preserved up to explicit perturbation bounds.


## Core mathematical idea

| Classical MDS | TMDS |
|---|---|
| Starts from pairwise distances | Starts from features and targets |
| Recovers coordinates via eigendecomposition of the centered Gram matrix | Learns a map B̂ so the feature-induced Gram approximates the target Gram |
| In-sample only | Out-of-sample by construction |

An assumption-free bound links prediction quality to geometric fidelity:

$$\|T_c T_c^\top - Z Z^\top\|_F \leq 2\|T_c\|_F \varepsilon + \varepsilon^2$$

where ε = ‖T_c − Y_c B̂‖_F. High R² on multivariate prediction implies faithful Gram recovery; no separate geometric loss is needed.

## Paper

The paper is in [`paper/`](paper/):

- **[TMDS_Theory.pdf](paper/TMDS_Theory.pdf)** - Full paper with appendices covering modular metric composition and rank truncation bounds.

## Benchmarks and Python demos

Install the small Python dependency set:

```bash
pip install -r requirements.txt
```

Run the simple Iris demo:

```bash
python examples/tmds_simple_demo.py
```

This fits the TMDS ridge map from Iris features to one-hot species targets, embeds samples with `Z = Yc @ B`, checks the exact metric identity `Yc M Yc^T = Z Z^T`, and clusters in the learned supervised geometry.

Run the larger Covertype scaling demo:

```bash
python examples/tmds_scaling_demo.py
```

This uses scikit-learn's 581k-row Forest Covertype dataset and reports runtime, represented dense-array memory, held-out accuracy, ARI, metric diagnostics, and the size of the explicit `n x n` matrix that TMDS avoids. The first run may download/cache the full source dataset; `--max-rows` caps computation after loading. For a quick smoke test, use:

```bash
python examples/tmds_scaling_demo.py --max-rows 50000
```

Run the Fashion-MNIST classification benchmark:

```bash
python examples/tmds_simple_classif_benchmark.py
```

This compares TMDS with scikit-learn LDA on a harder 70k-row image dataset and includes separate timing and represented dense-array memory rows for each method. The first run downloads Fashion-MNIST through scikit-learn/OpenML using the non-pandas ARFF parser; later runs use the local cache. `--max-rows` caps computation after loading. For a quick smoke test, use:

```bash
python examples/tmds_simple_classif_benchmark.py --max-rows 50000
```

All demos use ridge regularization `lambda = 1e-6` by default.

## Related problems

TMDS is most relevant to people looking for supervised metric learning, supervised multidimensional scaling, Mahalanobis metric learning, scalable clustering with labels, dimensionality reduction without pairwise distances, and MDS without an `n x n` distance matrix.

## Citation

```bibtex
@article{cerff2026tmds,
  title   = {No Distance Matrix Required: Multivariate Regression as Supervised Metric Learning},
  author  = {Cerff, Robert},
  year    = {2026},
  note    = {Available at \url{https://github.com/robcerff/TMDS}}
}
```

## License

[Apache 2.0](LICENSE)
