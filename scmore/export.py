from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from .logging_utils import get_logger

LOGGER = get_logger("export")

def stratified_cell_indices(obs: pd.DataFrame, max_cells: int, minimum: int, seed: int):
    n = len(obs)
    if n <= max_cells:
        return np.arange(n)
    columns = [x for x in ("sample_id", "gpt_cell_type") if x in obs]
    if not columns:
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n, max_cells, replace=False))
    groups = obs.reset_index(drop=True).groupby(columns, observed=True).indices
    sizes = pd.Series({key: len(ids) for key, ids in groups.items()})
    base = np.minimum(sizes, minimum)
    if base.sum() > max_cells:
        base = np.floor(sizes / sizes.sum() * max_cells).astype(int).clip(lower=1)
    remaining = max_cells - int(base.sum())
    capacity = sizes - base
    extra = np.floor(capacity / max(capacity.sum(), 1) * remaining).astype(int)
    allocation = (base + extra).clip(upper=sizes)
    leftover = max_cells - int(allocation.sum())
    for key in (sizes - allocation).sort_values(ascending=False).index:
        if leftover == 0:
            break
        take = min(leftover, int(sizes[key] - allocation[key]))
        allocation[key] += take
        leftover -= take
    rng = np.random.default_rng(seed)
    selected = [
        rng.choice(groups[key], int(allocation[key]), replace=False) for key in groups
    ]
    return np.sort(np.concatenate(selected))


def _write_sparse_long(
    x,
    rows,
    columns,
    row_name: str,
    column_name: str,
    expression_path: Path,
    index_path: Path,
    row_batch_size: int = 512,
) -> int:
    """Stream a sparse matrix to long-form Parquet without a full COO copy."""
    matrix = sparse.csr_matrix(x, copy=False)
    rows = np.asarray(rows, dtype=str)
    columns = np.asarray(columns, dtype=str)
    expression_tmp = expression_path.with_suffix(".parquet.tmp")
    index_tmp = index_path.with_suffix(".parquet.tmp")
    for path in (expression_tmp, index_tmp):
        if path.exists():
            path.unlink()
    expression_writer = index_writer = None
    written = 0
    try:
        for start in range(0, matrix.shape[0], row_batch_size):
            end = min(start + row_batch_size, matrix.shape[0])
            block = matrix[start:end].tocoo()
            if block.nnz == 0:
                continue
            frame = pd.DataFrame(
                {
                    row_name: rows[start:end][block.row],
                    column_name: columns[block.col],
                    "value": block.data.astype(np.float32, copy=False),
                }
            )
            table = pa.Table.from_pandas(frame, preserve_index=False)
            index_table = table.select([column_name, row_name])
            if expression_writer is None:
                expression_writer = pq.ParquetWriter(
                    expression_tmp, table.schema, compression="zstd"
                )
                index_writer = pq.ParquetWriter(
                    index_tmp, index_table.schema, compression="zstd"
                )
            expression_writer.write_table(table)
            index_writer.write_table(index_table)
            written += len(frame)
        if expression_writer is None:
            raise ValueError(f"{column_name} expression matrix contains no values")
    except BaseException:
        for writer in (expression_writer, index_writer):
            if writer is not None:
                writer.close()
        for path in (expression_tmp, index_tmp):
            if path.exists():
                path.unlink()
        raise
    else:
        expression_writer.close()
        index_writer.close()
        expression_tmp.replace(expression_path)
        index_tmp.replace(index_path)
    return written


def export_backend(
    dataset: str,
    mdata,
    out: Path,
    max_cells: int,
    min_per_stratum: int,
    seed: int,
    links: pd.DataFrame | None = None,
    motifs: pd.DataFrame | None = None,
    triplets: pd.DataFrame | None = None,
) -> None:
    for sub in ("meta", "embedding", "expression", "index", "regulatory"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    obsolete_wnn = out / "embedding/umap_wnn.parquet"
    if obsolete_wnn.exists():
        obsolete_wnn.unlink()
    rna, atac = mdata.mod["rna"], mdata.mod["atac"]
    selected = stratified_cell_indices(rna.obs, max_cells, min_per_stratum, seed)
    LOGGER.info(
        "downsampling original_cells=%d selected_cells=%d max_cells=%d seed=%d",
        rna.n_obs, len(selected), max_cells, seed,
    )
    cell_ids = rna.obs_names.astype(str).to_numpy()[selected]
    cells = rna.obs.iloc[selected].copy()
    cells.insert(0, "cell_id", cell_ids)
    cells.insert(0, "dataset", dataset)
    cells.reset_index(drop=True).to_parquet(out / "meta/cells.parquet", index=False)
    pd.DataFrame({
        "cell_id": rna.obs_names.astype(str),
        "selected": np.isin(np.arange(rna.n_obs), selected),
    }).to_parquet(out / "meta/downsampling_membership.parquet", index=False)
    for modality, adata in (("rna", rna), ("atac", atac)):
        if "X_umap" in adata.obsm:
            pd.DataFrame({
                "cell_id": cell_ids, "x": adata.obsm["X_umap"][selected, 0],
                "y": adata.obsm["X_umap"][selected, 1],
            }).to_parquet(out / f"embedding/umap_{modality}.parquet", index=False)
    rna_rows = _write_sparse_long(
        rna.X[selected], cell_ids, rna.var_names, "cell_id", "gene",
        out / "expression/rna_expr.parquet",
        out / "index/gene_to_cells.parquet",
    )
    atac_rows = _write_sparse_long(
        atac.X[selected], cell_ids, atac.var_names, "cell_id", "peak",
        out / "expression/atac_expr.parquet",
        out / "index/peak_to_cells.parquet",
    )
    LOGGER.info(
        "sparse_expression_export rna_rows=%d atac_rows=%d",
        rna_rows, atac_rows,
    )
    if links is not None:
        links.to_parquet(out / "regulatory/peak_gene_links.parquet", index=False)
    if motifs is not None:
        motifs.to_parquet(
            out / "regulatory/tf_peak_motif_hits.parquet", index=False
        )
    if triplets is not None:
        triplets.to_parquet(out / "regulatory/tf_peak_gene_triplets.parquet", index=False)
    (out / "meta/downsampling.json").write_text(json.dumps({
        "method": "sample_id x gpt_cell_type stratified random sampling",
        "seed": seed, "original_cells": rna.n_obs, "exported_cells": len(selected),
        "max_cells": max_cells, "minimum_per_stratum": min_per_stratum,
        "note": "active-score parquet is intentionally not downsampled",
    }, indent=2))
    LOGGER.info("backend_export_complete output=%s", out)
