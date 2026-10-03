from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from .logging_utils import get_logger


LOGGER = get_logger("context_means")

CONTEXT_FIELDS = [
    "sample_name",
    "biosource_cell_line",
    "biosource_cell_type",
    "gpt_cell_type",
    "biosource_disease",
    "experiment_treatment",
    "experiment_genotype",
    "experiment_group",
    "donor_id",
    "donor_sex",
    "donor_age",
    "donor_age_group",
]

MISSING_CONTEXT_VALUES = {
    "", "na", "nan", "none", "null", "unknown",
    "notavailable", "not available", "-", "--",
}


def _valid_context(values: pd.Series) -> np.ndarray:
    # Convert categoricals before filling: "" is generally not one of their
    # declared categories, so filling the categorical directly raises.
    normalized = values.astype(object).fillna("").astype(str).str.strip()
    return ~normalized.str.lower().isin(MISSING_CONTEXT_VALUES).to_numpy()


def _write_one(
    adata,
    feature_name: str,
    output: Path,
    feature_batch_size: int,
) -> int:
    temporary = output.with_suffix(".parquet.tmp")
    if temporary.exists():
        temporary.unlink()
    matrix = sparse.csr_matrix(adata.X, copy=False)
    feature_names = adata.var_names.astype(str).to_numpy()
    writer = None
    rows_written = 0
    try:
        for context_name in CONTEXT_FIELDS:
            if context_name not in adata.obs:
                LOGGER.warning(
                    "context_field_missing modality=%s field=%s",
                    feature_name, context_name,
                )
                continue
            values = adata.obs[context_name]
            valid = _valid_context(values)
            if not valid.any():
                LOGGER.info(
                    "context_field_all_missing modality=%s field=%s",
                    feature_name, context_name,
                )
                continue
            labels = values.iloc[np.flatnonzero(valid)].astype(str).str.strip()
            codes, groups = pd.factorize(labels, sort=True)
            valid_rows = np.flatnonzero(valid)
            n_cells = np.bincount(codes, minlength=len(groups)).astype(np.int64)
            indicator = sparse.csr_matrix(
                (
                    np.ones(len(valid_rows), dtype=np.float32),
                    (codes, valid_rows),
                ),
                shape=(len(groups), adata.n_obs),
            )
            for start in range(0, adata.n_vars, feature_batch_size):
                end = min(start + feature_batch_size, adata.n_vars)
                block = matrix[:, start:end].copy()
                sums = (indicator @ block).tocoo()
                if sums.nnz == 0:
                    continue
                block.data = np.ones(block.nnz, dtype=np.float32)
                nnz = (indicator @ block).tocsr()
                counts = np.asarray(nnz[sums.row, sums.col]).ravel().astype(np.int64)
                frame = pd.DataFrame(
                    {
                        "context_name": context_name,
                        "context_value": groups.to_numpy(dtype=str)[sums.row],
                        feature_name: feature_names[start:end][sums.col],
                        "mean_value": (
                            sums.data.astype(np.float64) / n_cells[sums.row]
                        ),
                        "n_cells": n_cells[sums.row],
                        "nnz": counts,
                    }
                )
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(
                        temporary, table.schema, compression="zstd"
                    )
                writer.write_table(table)
                rows_written += len(frame)
        if writer is None:
            raise ValueError(
                f"no usable context means generated for {feature_name}"
            )
    except BaseException:
        if writer is not None:
            writer.close()
            writer = None
        if temporary.exists():
            temporary.unlink()
        raise
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output)
    LOGGER.info(
        "context_means_complete feature=%s rows=%d output=%s",
        feature_name, rows_written, output,
    )
    return rows_written


def build_context_means(
    mdata,
    dataset_dir: Path,
    feature_batch_size: int = 5_000,
) -> dict[str, int]:
    """Build full-data, sparse, zero-aware context means for RNA and ATAC."""
    if feature_batch_size < 1:
        raise ValueError("feature_batch_size must be positive")
    expression_dir = dataset_dir / "expression"
    gene_rows = _write_one(
        mdata.mod["rna"],
        "gene",
        expression_dir / "gene_mean_context.parquet",
        feature_batch_size,
    )
    peak_rows = _write_one(
        mdata.mod["atac"],
        "peak",
        expression_dir / "peak_mean_context.parquet",
        feature_batch_size,
    )
    return {"gene_rows": gene_rows, "peak_rows": peak_rows}
