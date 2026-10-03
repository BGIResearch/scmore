from __future__ import annotations

import gc
import os
from pathlib import Path

import mudata as md
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from .data_processing import PeakAnnotator, load_dataset, normalize_peak_names
from .logging_utils import get_logger

LOGGER = get_logger("preprocess")

def _detected(x) -> np.ndarray:
    return np.asarray(x.getnnz(axis=0)).ravel() if sparse.issparse(x) else np.count_nonzero(x, axis=0)


def _sync_sample_to_mudata(mdata) -> None:
    """Validate modality sample labels and preserve them at MuData level."""
    rna, atac = mdata.mod["rna"], mdata.mod["atac"]
    if "sample" not in rna.obs or "sample" not in atac.obs:
        raise ValueError("rna.obs and atac.obs must both contain sample")
    rna_sample = rna.obs["sample"].astype(str)
    atac_sample = atac.obs["sample"].astype(str).reindex(rna.obs_names)
    if atac_sample.isna().any() or not np.array_equal(
        rna_sample.to_numpy(), atac_sample.to_numpy()
    ):
        raise ValueError("RNA and ATAC sample labels differ")
    values = rna_sample.reindex(mdata.obs_names).to_numpy(dtype=object)
    if pd.isna(values).any():
        raise ValueError("MuData cells could not be mapped to sample labels")
    mdata.obs["sample"] = pd.Categorical(values)


def atomic_write_h5mu(path: Path, mdata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    building = path.with_name(
        f".{path.stem}.building.{os.getpid()}{path.suffix}"
    )
    md.write_h5mu(building, mdata)
    building.replace(path)


def load_filter_annotate(
    data_path: Path,
    sample_names: list[str],
    sample_files: list[str] | None,
    gtf_path: Path,
    feature_fraction: float,
):
    """Read, QC, merge, filter, and annotate without the legacy pipeline."""
    mdata = load_dataset(
        data_path, sample_names, sample_files=sample_files,
        feature_fraction=feature_fraction,
    )
    rna, atac = mdata.mod["rna"], mdata.mod["atac"]
    if not np.array_equal(rna.obs_names, atac.obs_names):
        raise ValueError("RNA and ATAC cell orders differ")
    keep_rna = _detected(rna.X) > rna.n_obs * feature_fraction
    keep_atac = _detected(atac.X) > atac.n_obs * feature_fraction
    LOGGER.info(
        "feature_filter fraction=%g genes_before=%d genes_after=%d "
        "peaks_before=%d peaks_after=%d",
        feature_fraction, rna.n_vars, int(keep_rna.sum()),
        atac.n_vars, int(keep_atac.sum()),
    )
    # Avoid duplicating both full matrices. The previous implementation made
    # one filtered MuData copy and then another result copy.
    rna._inplace_subset_var(keep_rna)
    atac._inplace_subset_var(keep_atac)
    mdata.update()
    _sync_sample_to_mudata(mdata)
    atac = normalize_peak_names(atac)
    # Rebuild the lightweight container after normalization. If normalization
    # both renames peaks and collapses duplicate columns, updating the old
    # MuData container cannot reconcile its stale global var index.
    uns = mdata.uns
    mdata = md.MuData({"rna": rna, "atac": atac})
    mdata.uns = uns
    _sync_sample_to_mudata(mdata)
    peak_frame = atac.var_names.to_series().str.extract(
        r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$"
    )
    if peak_frame.isna().any(axis=None):
        raise ValueError("one or more ATAC peak names are not chrom:start-end")
    peak_frame[["start", "end"]] = peak_frame[["start", "end"]].astype(np.int64)
    annotation = PeakAnnotator(gtf_path).annotate(peak_frame)
    annotation.index = annotation.index.astype(str)
    valid = annotation["gene"].fillna("").astype(str).str.strip().ne("")
    annotation = annotation.loc[valid]
    keep_peaks = atac.var_names.astype(str).isin(annotation.index)
    LOGGER.info(
        "peak_annotation annotated_pairs=%d retained_peaks=%d",
        len(annotation), int(keep_peaks.sum()),
    )
    atac._inplace_subset_var(keep_peaks)
    mdata.uns["peak_annotation"] = annotation.loc[
        annotation.index.isin(atac.var_names.astype(str))
    ].copy()
    mdata.update()
    _sync_sample_to_mudata(mdata)
    for adata in mdata.mod.values():
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    gc.collect()
    return mdata


def cluster_and_markers(
    mdata,
    marker_csv: Path,
    n_top_genes: int = 3000,
    n_pcs: int = 20,
    resolution: float = 1.0,
) -> None:
    rna = mdata.mod["rna"]
    sc.pp.highly_variable_genes(rna, n_top_genes=min(n_top_genes, rna.n_vars), subset=False)
    analysis = rna[:, rna.var["highly_variable"].to_numpy()].copy()
    sc.pp.scale(analysis, zero_center=False)
    sc.pp.pca(
        analysis,
        n_comps=min(n_pcs, analysis.n_vars - 1, analysis.n_obs - 1),
        zero_center=False,
    )
    sc.pp.neighbors(analysis)
    sc.tl.leiden(
        analysis, resolution=resolution, key_added="leiden", flavor="igraph",
        n_iterations=2, directed=False,
    )
    rna.obs["leiden"] = analysis.obs["leiden"].astype(str).to_numpy()
    sc.tl.rank_genes_groups(
        rna, groupby="leiden", method="wilcoxon", use_raw=False,
        pts=True, key_added="rank_genes_groups_wilcoxon",
    )
    markers = sc.get.rank_genes_groups_df(rna, group=None, key="rank_genes_groups_wilcoxon")
    marker_csv.parent.mkdir(parents=True, exist_ok=True)
    markers.to_csv(marker_csv, index=False)
    for adata in [mdata, *mdata.mod.values()]:
        adata.obs["leiden"] = rna.obs["leiden"].reindex(adata.obs_names).astype(str).to_numpy()
