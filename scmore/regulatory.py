from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from scipy.stats import rankdata, t as student_t

from .config import Thresholds
from .logging_utils import get_logger


GROUP_COLUMNS = ["project_id", "sample_id", "gpt_cell_type"]
LOGGER = get_logger("regulatory")


def _pseudobulk(
    adata, features: pd.Index, min_cells: int,
    *, allow_pseudo_replicates: bool = False,
    pseudo_replicate_min_cells: int = 50,
    pseudo_replicate_target: int = 5,
    random_seed: int = 0,
):
    frame = adata.obs[GROUP_COLUMNS].astype(str)
    codes0, groups0 = pd.factorize(pd.MultiIndex.from_frame(frame), sort=True)
    sizes0 = np.bincount(codes0)
    keep_group = sizes0 >= min_cells
    keep_cell = keep_group[codes0]
    remap = np.full(len(groups0), -1, dtype=int)
    remap[np.flatnonzero(keep_group)] = np.arange(keep_group.sum())
    codes = remap[codes0[keep_cell]]
    kept_groups = groups0[keep_group].to_frame(index=False)
    kept_groups.columns = GROUP_COLUMNS
    mode = "biological"
    replicate_ids = np.zeros(len(kept_groups), dtype=int)
    if allow_pseudo_replicates and len(kept_groups) < 3 and len(kept_groups):
        desired = max(2, int(np.ceil(pseudo_replicate_target / len(kept_groups))))
        rng = np.random.default_rng(random_seed)
        split_codes = np.full(len(codes), -1, dtype=int)
        split_groups, split_reps, next_code = [], [], 0
        for group_id in range(len(kept_groups)):
            cells = np.flatnonzero(codes == group_id)
            n_parts = min(desired, len(cells) // pseudo_replicate_min_cells)
            n_parts = max(1, n_parts)
            shuffled = rng.permutation(cells)
            for replicate_id, members in enumerate(np.array_split(shuffled, n_parts)):
                split_codes[members] = next_code
                split_groups.append(kept_groups.iloc[group_id].copy())
                split_reps.append(replicate_id)
                next_code += 1
        if next_code >= 3:
            codes = split_codes
            kept_groups = pd.DataFrame(split_groups).reset_index(drop=True)
            replicate_ids = np.asarray(split_reps, dtype=int)
            mode = "within_sample_split"
    sizes = np.bincount(codes)
    indicator = sparse.csr_matrix(
        (
            (1.0 / sizes[codes]).astype(np.float32),
            (codes, np.arange(len(codes))),
        ),
        shape=(len(sizes), len(codes)),
    )
    pos = adata.var_names.get_indexer(features)
    if np.any(pos < 0):
        raise ValueError(f"features absent from matrix: {features[pos < 0].tolist()[:10]}")
    values = sparse.csr_matrix(adata.X)[keep_cell][:, pos]
    means = (indicator @ values).toarray().astype(np.float32)
    groups = kept_groups
    groups["n_cells"] = sizes
    groups["pseudobulk_mode"] = mode
    groups["pseudo_replicate"] = replicate_ids
    return means, groups


def _rank_columns(x: np.ndarray):
    ranks = np.apply_along_axis(rankdata, 0, x, method="average")
    ranks -= ranks.mean(axis=0, keepdims=True)
    return ranks, np.sqrt(np.square(ranks).sum(axis=0))


def compute_peak_gene_links(
    mdata, output: Path, min_cells: int, *,
    allow_pseudo_replicates: bool = True,
    pseudo_replicate_min_cells: int = 50,
    pseudo_replicate_target: int = 5,
    random_seed: int = 0,
) -> pd.DataFrame:
    rna, atac = mdata.mod["rna"], mdata.mod["atac"]
    ann = mdata.uns["peak_annotation"].copy()
    ann.index = ann.index.astype(str)
    pairs = ann.reset_index(names="peak")[["peak", "gene"]]
    pairs["gene"] = pairs["gene"].astype(str)
    pairs = pairs[
        pairs["peak"].isin(atac.var_names) & pairs["gene"].isin(rna.var_names)
    ].drop_duplicates().reset_index(drop=True)
    genes, peaks = pd.Index(pairs["gene"].unique()), pd.Index(pairs["peak"].unique())
    pb_kwargs = dict(
        allow_pseudo_replicates=allow_pseudo_replicates,
        pseudo_replicate_min_cells=pseudo_replicate_min_cells,
        pseudo_replicate_target=pseudo_replicate_target,
        random_seed=random_seed,
    )
    rna_pb, groups = _pseudobulk(rna, genes, min_cells, **pb_kwargs)
    atac_pb, atac_groups = _pseudobulk(atac, peaks, min_cells, **pb_kwargs)
    group_identity = GROUP_COLUMNS + ["pseudobulk_mode", "pseudo_replicate"]
    if not groups[group_identity].equals(atac_groups[group_identity]):
        raise ValueError("RNA and ATAC pseudobulk groups differ")
    rr, rn = _rank_columns(rna_pb)
    ar, an = _rank_columns(atac_pb)
    gene_pos = {x: i for i, x in enumerate(genes)}
    peak_pos = {x: i for i, x in enumerate(peaks)}
    rho = np.empty(len(pairs), dtype=float)
    for gene, ids0 in pairs.groupby("gene").groups.items():
        ids = np.asarray(list(ids0))
        pi = np.array([peak_pos[x] for x in pairs.loc[ids, "peak"]])
        numerator = ar[:, pi].T @ rr[:, gene_pos[gene]]
        denominator = an[pi] * rn[gene_pos[gene]]
        rho[ids] = np.divide(numerator, denominator, out=np.full(len(ids), np.nan), where=denominator > 0)
    n = len(groups)
    pvalue = np.full(len(rho), np.nan)
    valid = np.isfinite(rho)
    stat = rho[valid] * np.sqrt((n - 2) / np.maximum(1 - rho[valid] ** 2, np.finfo(float).tiny))
    pvalue[valid] = 2 * student_t.sf(np.abs(stat), df=n - 2)
    order = np.argsort(pvalue[valid])
    q = np.full(valid.sum(), np.nan)
    ranked = pvalue[valid][order] * valid.sum() / np.arange(1, valid.sum() + 1)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    qvalue = np.full(len(rho), np.nan)
    qvalue[valid] = q
    result = pairs.assign(
        spearman=rho, pvalue=pvalue, qvalue_bh=qvalue,
        n_pseudobulks=n,
        pseudobulk_mode=groups["pseudobulk_mode"].iloc[0],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    groups.to_csv(output.with_suffix(".groups.csv"), index=False)
    LOGGER.info(
        "peak_gene_links_complete pairs=%d pseudobulks=%d output=%s",
        len(result), len(groups), output,
    )
    return result


def build_triplets(
    mdata,
    links: pd.DataFrame,
    motif_source: Path | pd.DataFrame,
    output: Path,
    thresholds: Thresholds,
    *,
    tf_expression_mean: float | None = None,
    tf_expression_max: float | None = None,
    motif_pvalue: float | None = None,
    tf_gene_r: float | None = None,
    tf_gene_fdr: float | None = None,
    tf_gene_absolute: bool = False,
    allow_pseudo_replicates: bool = True,
    random_seed: int = 0,
) -> pd.DataFrame:
    motifs = (
        motif_source
        if isinstance(motif_source, pd.DataFrame)
        else pd.read_parquet(motif_source)
    )
    motifs = motifs.rename(
        columns={"TF_Name": "tf", "p-value": "motif_pvalue", "pvalue": "motif_pvalue"}
    )
    required = {"tf", "peak", "motif_pvalue"}
    if required - set(motifs):
        raise ValueError(f"motif table missing: {sorted(required-set(motifs))}")
    rna = mdata.mod["rna"]
    tfs = pd.Index(sorted(set(motifs["tf"].astype(str)) & set(rna.var_names)))
    tf_mean = thresholds.tf_pseudobulk_mean if tf_expression_mean is None else tf_expression_mean
    tf_max = thresholds.tf_pseudobulk_max if tf_expression_max is None else tf_expression_max
    effective_motif_pvalue = thresholds.motif_pvalue if motif_pvalue is None else motif_pvalue
    final_tf_gene_r = thresholds.tf_gene_r if tf_gene_r is None else tf_gene_r
    tf_pb, tf_groups = _pseudobulk(
        rna, tfs, thresholds.min_pseudobulk_cells,
        allow_pseudo_replicates=allow_pseudo_replicates,
        random_seed=random_seed,
    )
    metrics = pd.DataFrame({
        "tf": tfs,
        "pseudobulk_expression_mean": tf_pb.mean(axis=0),
        "pseudobulk_expression_max": tf_pb.max(axis=0),
    })
    eligible = metrics[
        (metrics["pseudobulk_expression_mean"] > tf_mean)
        & (metrics["pseudobulk_expression_max"] > tf_max)
    ]
    selected_links = links[
        (links["qvalue_bh"] < thresholds.peak_gene_fdr)
        & (links["spearman"] > thresholds.peak_gene_r)
    ]
    selected_motifs = motifs[
        (motifs["motif_pvalue"] < effective_motif_pvalue)
        & motifs["tf"].isin(eligible["tf"])
    ].sort_values("motif_pvalue").drop_duplicates(["tf", "peak"])
    triplets = selected_links.merge(selected_motifs, on="peak").merge(eligible, on="tf")
    if triplets.empty:
        result = triplets.assign(
            tf_gene_spearman=pd.Series(index=triplets.index, dtype=float)
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output, index=False)
        metrics.to_csv(output.with_suffix(".tf_expression.csv"), index=False)
        output.with_suffix(".thresholds.json").write_text(
            json.dumps(thresholds.__dict__, indent=2)
        )
        LOGGER.warning(
            "triplets_empty selected_links=%d selected_motifs=%d output=%s",
            len(selected_links), len(selected_motifs), output,
        )
        return result
    features = pd.Index(pd.unique(pd.concat([triplets["tf"], triplets["gene"]], ignore_index=True)))
    pb, groups = _pseudobulk(
        rna, features, thresholds.min_pseudobulk_cells,
        allow_pseudo_replicates=allow_pseudo_replicates,
        random_seed=random_seed,
    )
    ranks, norms = _rank_columns(pb)
    lookup = {x: i for i, x in enumerate(features)}
    pairs = triplets[["tf", "gene"]].drop_duplicates()
    corr = []
    for tf, gene in pairs.itertuples(index=False):
        ti, gi = lookup[tf], lookup[gene]
        denom = norms[ti] * norms[gi]
        corr.append(np.nan if denom == 0 else float(ranks[:, ti] @ ranks[:, gi] / denom))
    pairs["tf_gene_spearman"] = corr
    n = len(groups)
    pvalue = np.full(len(pairs), np.nan)
    rho = pairs["tf_gene_spearman"].to_numpy(dtype=float)
    valid = np.isfinite(rho) & (n > 2)
    stat = rho[valid] * np.sqrt(
        (n - 2) / np.maximum(1 - rho[valid] ** 2, np.finfo(float).eps)
    )
    pvalue[valid] = 2 * student_t.sf(np.abs(stat), df=n - 2)
    qvalue = np.full(len(pairs), np.nan)
    if valid.any():
        order = np.argsort(pvalue[valid])
        adjusted = np.empty(valid.sum(), dtype=float)
        ranked = pvalue[valid][order] * valid.sum() / np.arange(1, valid.sum() + 1)
        adjusted[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
        qvalue[valid] = adjusted
    pairs["tf_gene_pvalue"] = pvalue
    pairs["tf_gene_qvalue_bh"] = qvalue
    result = triplets.merge(pairs, on=["tf", "gene"])
    if tf_gene_absolute:
        keep = result["tf_gene_spearman"].abs() > final_tf_gene_r
    else:
        keep = result["tf_gene_spearman"] > final_tf_gene_r
    if tf_gene_fdr is not None:
        keep &= result["tf_gene_qvalue_bh"] < tf_gene_fdr
    result = result[keep].copy()
    result["pseudobulk_mode"] = groups["pseudobulk_mode"].iloc[0]
    result["effective_tf_mean_threshold"] = tf_mean
    result["effective_tf_max_threshold"] = tf_max
    result["effective_motif_pvalue"] = effective_motif_pvalue
    result["effective_tf_gene_r"] = final_tf_gene_r
    result["effective_tf_gene_fdr"] = tf_gene_fdr
    result["regulation_direction"] = np.where(
        result["tf_gene_spearman"] >= 0, "activating", "repressive"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    metrics.to_csv(output.with_suffix(".tf_expression.csv"), index=False)
    output.with_suffix(".thresholds.json").write_text(json.dumps({
        **thresholds.__dict__,
        "effective_tf_pseudobulk_mean": tf_mean,
        "effective_tf_pseudobulk_max": tf_max,
        "effective_motif_pvalue": effective_motif_pvalue,
        "effective_tf_gene_r": final_tf_gene_r,
        "effective_tf_gene_fdr": tf_gene_fdr,
        "tf_gene_absolute": tf_gene_absolute,
        "pseudobulk_mode": groups["pseudobulk_mode"].iloc[0],
        "n_pseudobulks": n,
    }, indent=2))
    LOGGER.info(
        "triplets_complete selected_links=%d selected_motifs=%d triplets=%d output=%s",
        len(selected_links), len(selected_motifs), len(result), output,
    )
    return result


def compute_active_scores(
    dataset: str, mdata, triplets: pd.DataFrame, output: Path,
    min_cells: int, batch_size: int = 200_000,
) -> None:
    if triplets.empty:
        output.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "dataset", *GROUP_COLUMNS, "pseudobulk_n_cells", "tf", "peak",
            "gene", "spearman", "qvalue_bh", "tf_gene_spearman",
            "motif_pvalue", "tf_pseudobulk_mean", "peak_pseudobulk_mean",
            "gene_pseudobulk_mean", "active_score", "weighted_active_score",
        ]
        pd.DataFrame({column: pd.Series(dtype="object") for column in columns}).to_parquet(
            output, index=False
        )
        LOGGER.warning("active_scores_empty output=%s", output)
        return
    rna, atac = mdata.mod["rna"], mdata.mod["atac"]
    features = pd.Index(pd.unique(pd.concat([triplets["tf"], triplets["gene"]], ignore_index=True)))
    peaks = pd.Index(triplets["peak"].unique())
    rna_pb, groups = _pseudobulk(rna, features, min_cells)
    atac_pb, groups2 = _pseudobulk(atac, peaks, min_cells)
    if not groups[GROUP_COLUMNS].equals(groups2[GROUP_COLUMNS]):
        raise ValueError("RNA and ATAC pseudobulk groups differ")
    def percentiles(x):
        result = np.apply_along_axis(rankdata, 0, x) / x.shape[0]
        result[x == 0] = 0
        return result.astype(np.float32)
    rrank, arank = percentiles(rna_pb), percentiles(atac_pb)
    fpos, ppos = {x: i for i, x in enumerate(features)}, {x: i for i, x in enumerate(peaks)}
    ti = triplets["tf"].map(fpos).to_numpy()
    gi = triplets["gene"].map(fpos).to_numpy()
    pi = triplets["peak"].map(ppos).to_numpy()
    writer = None
    try:
        for group_id, group in groups.iterrows():
            for start in range(0, len(triplets), batch_size):
                sl = slice(start, min(start + batch_size, len(triplets)))
                active = np.cbrt(rrank[group_id, ti[sl]] * arank[group_id, pi[sl]] * rrank[group_id, gi[sl]])
                frame = triplets.iloc[sl][["tf", "peak", "gene", "spearman", "qvalue_bh", "tf_gene_spearman", "motif_pvalue"]].copy()
                frame.insert(0, "dataset", dataset)
                for col in GROUP_COLUMNS:
                    frame[col] = group[col]
                frame["pseudobulk_n_cells"] = int(group["n_cells"])
                frame["tf_pseudobulk_mean"] = rna_pb[group_id, ti[sl]]
                frame["peak_pseudobulk_mean"] = atac_pb[group_id, pi[sl]]
                frame["gene_pseudobulk_mean"] = rna_pb[group_id, gi[sl]]
                frame["active_score"] = active
                frame["weighted_active_score"] = active * np.sqrt(frame["spearman"] * frame["tf_gene_spearman"])
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(output, table.schema, compression="zstd")
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("no active-score rows generated")
    LOGGER.info(
        "active_scores_complete groups=%d triplets=%d output=%s",
        len(groups), len(triplets), output,
    )
