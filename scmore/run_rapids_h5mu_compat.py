#!/usr/bin/env python
"""Safely add RNA/ATAC RAPIDS embeddings while preserving H5MU recarrays.

This is the local, WNN-free replacement for
``case/rep/run_rapids_h5mu_compat.py``.  The parent process protects the
rank-genes recarray, launches the selected RAPIDS environment, validates the
result, and replaces the original only after every check succeeds.
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
from pathlib import Path
import subprocess
import sys

import h5py


DEFAULT_RAPIDS_ENV = Path("/path/to/rapids_singlecell_env")
GPU_PCA_MAX_FEATURES = 32_768
RANK_GROUP = "mod/rna/uns/rank_genes_groups_wilcoxon"


def _to_cpu(value, converter):
    try:
        converted = converter(value)
    except (TypeError, AttributeError):
        converted = value
    return converted.copy() if hasattr(converted, "copy") else converted


def _remove_null_encoded_nodes(handle: h5py.File) -> list[str]:
    """Drop optional None values written by newer anndata releases.

    Older anndata versions cannot read the ``null`` encoding at all. These
    nodes only represent absent optional parameters, so omitting them retains
    the same semantics and keeps the H5MU readable in both environments.
    """
    paths: list[str] = []

    def collect(name: str, item) -> None:
        encoding = item.attrs.get("encoding-type")
        if isinstance(encoding, bytes):
            encoding = encoding.decode()
        if encoding == "null":
            paths.append(name)

    handle.visititems(collect)
    for name in sorted(paths, key=lambda value: value.count("/"), reverse=True):
        del handle[name]
    return paths


def run_embedding_worker(
    path: Path,
    n_pcs: int,
    n_neighbors: int,
    random_state: int,
    neighbors_algorithm: str,
) -> None:
    """Executed inside the RAPIDS environment; never computes WNN."""
    import mudata as md
    import numpy as np
    import rapids_singlecell as rsc
    from rapids_singlecell.get import X_to_CPU, anndata_to_GPU

    mdata = md.read_h5mu(path)
    rna, atac = mdata.mod["rna"], mdata.mod["atac"]
    if not rna.obs_names.equals(atac.obs_names):
        raise ValueError("RNA and ATAC cells/order do not match")
    if "highly_variable" not in rna.var:
        raise KeyError("rna.var['highly_variable'] is missing")
    if "peak_annotation" not in mdata.uns:
        raise KeyError("mdata.uns['peak_annotation'] is missing")

    hvg_mask = rna.var["highly_variable"].astype(bool).to_numpy()
    if not hvg_mask.any():
        raise ValueError("RNA contains no highly variable genes")
    hvg_genes = set(rna.var_names[hvg_mask])
    annotation = mdata.uns["peak_annotation"]
    linked_peaks = set(annotation.index[annotation["gene"].isin(hvg_genes)])
    peak_mask = np.asarray(atac.var_names.isin(linked_peaks))
    if not peak_mask.any():
        raise ValueError("no ATAC peaks link to RNA HVGs")
    atac.var["peak_in_rna_hvg"] = peak_mask

    for name, target, mask in (
        ("rna", rna, hvg_mask),
        ("atac", atac, peak_mask),
    ):
        # Construct one modality subset at a time. A tuple containing both
        # copies evaluates eagerly and can double peak host memory.
        source = target[:, mask].copy()
        n_components = min(n_pcs, source.n_obs - 1, source.n_vars - 1)
        if n_components < 2:
            raise ValueError(f"{name}: insufficient dimensions for PCA")
        if source.n_vars > GPU_PCA_MAX_FEATURES:
            # RAPIDS covariance PCA becomes quadratic in feature count and
            # cuSOLVER can reject matrices above its high-dimensional boundary.
            # Randomized TruncatedSVD works directly on host CSR without
            # constructing a feature-by-feature covariance matrix. Only PCA
            # falls back to CPU; neighbors and UMAP remain on the GPU.
            from sklearn.decomposition import TruncatedSVD
            from threadpoolctl import threadpool_limits

            cpu_threads = min(16, os.cpu_count() or 1)
            print(
                f"[hybrid] {name} features={source.n_vars} exceeds "
                f"{GPU_PCA_MAX_FEATURES}; CPU randomized TruncatedSVD "
                f"threads={cpu_threads}, GPU neighbors/UMAP",
                flush=True,
            )
            svd = TruncatedSVD(
                n_components=n_components,
                algorithm="randomized",
                n_iter=5,
                n_oversamples=10,
                random_state=random_state,
            )
            with threadpool_limits(limits=cpu_threads):
                source.obsm["X_pca"] = svd.fit_transform(source.X).astype(
                    np.float32, copy=False
                )
            source.uns["pca"] = {
                "params": {
                    "zero_center": False,
                    "use_highly_variable": False,
                    "algorithm": "randomized_truncated_svd_cpu",
                },
                "variance": svd.explained_variance_.astype(np.float32),
                "variance_ratio": svd.explained_variance_ratio_.astype(
                    np.float32
                ),
            }
            anndata_to_GPU(source)
        else:
            anndata_to_GPU(source)
            rsc.pp.pca(
                source,
                n_comps=n_components,
                random_state=random_state,
                zero_center=False,
            )
        rsc.pp.neighbors(
            source,
            n_neighbors=n_neighbors,
            n_pcs=n_components,
            algorithm=neighbors_algorithm,
            random_state=random_state,
        )
        rsc.tl.umap(source, random_state=random_state)
        target.obsm["X_pca"] = np.asarray(
            _to_cpu(source.obsm["X_pca"], X_to_CPU), dtype=np.float32
        )
        target.obsm["X_umap"] = np.asarray(
            _to_cpu(source.obsm["X_umap"], X_to_CPU), dtype=np.float32
        )
        target.obsp["distances"] = _to_cpu(source.obsp["distances"], X_to_CPU)
        target.obsp["connectivities"] = _to_cpu(
            source.obsp["connectivities"], X_to_CPU
        )
        for key in ("pca", "neighbors", "umap"):
            if key in source.uns:
                target.uns[key] = source.uns[key].copy()
        del source
        gc.collect()
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except ImportError:
            pass

    # Remove artifacts if the input came from an older WNN-enabled run.
    for key in ("wnn_umap", "wnn_weights"):
        if key in mdata.obsm:
            del mdata.obsm[key]
    for key in ("wnn_connectivities", "wnn_distances"):
        if key in mdata.obsp:
            del mdata.obsp[key]
    if "wnn" in mdata.uns:
        del mdata.uns["wnn"]

    building = path.with_name(f"{path.stem}.building{path.suffix}")
    md.write_h5mu(building, mdata)
    building.replace(path)


def _runtime_environment(rapids_env: Path, gpu: str) -> dict[str, str]:
    sklearn_gomp = glob.glob(
        str(rapids_env / "lib/python*/site-packages/scikit_learn.libs/libgomp-*.so*")
    )
    env_gomp = rapids_env / "lib/libgomp.so.1"
    if not sklearn_gomp:
        raise FileNotFoundError(f"scikit-learn libgomp not found below {rapids_env}")
    if not env_gomp.exists():
        raise FileNotFoundError(env_gomp)
    env = os.environ.copy()
    env.update(
        {
            "LD_PRELOAD": f"{sklearn_gomp[0]}:{env_gomp}",
            "NUMBA_CACHE_DIR": "/tmp/scmore_v2_numba_cache",
            "MPLCONFIGDIR": "/tmp/scmore_v2_mpl_cache",
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            "CUDA_VISIBLE_DEVICES": str(gpu),
        }
    )
    return env


def run_one(
    path: Path,
    gpu: str = "0",
    rapids_env: Path = DEFAULT_RAPIDS_ENV,
    n_pcs: int = 50,
    n_neighbors: int = 20,
    random_state: int = 0,
    neighbors_algorithm: str = "cagra",
) -> None:
    path = path.resolve()
    rapids_env = rapids_env.resolve()
    temporary = path.with_name(
        f"{path.stem}.rapids_input.{os.getpid()}{path.suffix}"
    )
    building = temporary.with_name(f"{temporary.stem}.building{temporary.suffix}")
    if not path.exists():
        raise FileNotFoundError(path)

    subprocess.run(
        ["cp", "--reflink=auto", "--preserve=mode,timestamps", str(path), str(temporary)],
        check=True,
    )
    try:
        with h5py.File(temporary, "r+") as handle:
            rank_exists = RANK_GROUP in handle
            if rank_exists:
                del handle[RANK_GROUP]

        command = [
            str(rapids_env / "bin/python"),
            str(Path(__file__).resolve()),
            "--worker",
            "--n-pcs", str(n_pcs),
            "--n-neighbors", str(n_neighbors),
            "--random-state", str(random_state),
            "--neighbors-algorithm", neighbors_algorithm,
            str(temporary),
        ]
        subprocess.run(
            command, check=True, env=_runtime_environment(rapids_env, gpu)
        )

        if rank_exists:
            with h5py.File(path, "r") as source, h5py.File(temporary, "r+") as result:
                source.copy(RANK_GROUP, result["mod/rna/uns"])

        with h5py.File(temporary, "r+") as result:
            removed_nulls = _remove_null_encoded_nodes(result)
            if removed_nulls:
                print(
                    f"[compat] removed null-encoded nodes: {removed_nulls}",
                    flush=True,
                )

        with h5py.File(temporary, "r") as result:
            n_obs = result["mod/rna/obs/_index"].shape[0]
            expected = (n_obs, 2)
            checks = {
                "rna_pca": result["mod/rna/obsm/X_pca"].shape,
                "rna_umap": result["mod/rna/obsm/X_umap"].shape,
                "atac_pca": result["mod/atac/obsm/X_pca"].shape,
                "atac_umap": result["mod/atac/obsm/X_umap"].shape,
            }
            if checks["rna_pca"][0] != n_obs or checks["atac_pca"][0] != n_obs:
                raise ValueError(f"PCA validation failed: {checks}")
            if checks["rna_umap"] != expected or checks["atac_umap"] != expected:
                raise ValueError(f"UMAP validation failed: {checks}")
            for old in (
                "obsm/wnn_umap", "obsm/wnn_weights",
                "obsp/wnn_distances", "obsp/wnn_connectivities", "uns/wnn",
            ):
                if old in result:
                    raise ValueError(f"obsolete WNN artifact remains: {old}")
            if rank_exists and RANK_GROUP not in result:
                raise ValueError("rank_genes_groups recarray was not restored")
            print(f"[validated] {path.name}: {checks}", flush=True)
        temporary.replace(path)
        print(f"[replaced] {path}", flush=True)
    except BaseException:
        print(
            f"[failed safely] original retained; temporary file: {temporary}",
            file=sys.stderr,
            flush=True,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--rapids-env", type=Path, default=DEFAULT_RAPIDS_ENV)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--neighbors-algorithm", default="cagra")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if len(args.inputs) != 1:
            parser.error("--worker accepts exactly one input")
        run_embedding_worker(
            args.inputs[0], args.n_pcs, args.n_neighbors, args.random_state,
            args.neighbors_algorithm,
        )
        return
    for path in args.inputs:
        run_one(
            path, args.gpu, args.rapids_env, args.n_pcs, args.n_neighbors,
            args.random_state, args.neighbors_algorithm,
        )


if __name__ == "__main__":
    main()
