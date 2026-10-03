from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mudata as md
import numpy as np
import pandas as pd
from scipy import sparse


GROUP_COLUMNS = ["project_id", "sample_id", "gpt_cell_type"]
KEY_COLUMNS = ["tf", "peak", "gene"]
LOGGER = logging.getLogger(__name__)


def _positive_rows(matrix: sparse.csc_matrix, column: int) -> np.ndarray:
    start, stop = matrix.indptr[column : column + 2]
    rows = matrix.indices[start:stop]
    values = matrix.data[start:stop]
    return rows[values > 0]


def add_active_cell_fraction(h5mu: Path, active_scores: Path, output: Path) -> None:
    scores = pd.read_parquet(active_scores)
    required = set(KEY_COLUMNS + GROUP_COLUMNS)
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"active-score table is missing columns: {sorted(missing)}")

    triplets = scores[KEY_COLUMNS].drop_duplicates().reset_index(drop=True)
    mdata = md.read_h5mu(h5mu, backed="r")
    try:
        rna, atac = mdata.mod["rna"], mdata.mod["atac"]
        if not rna.obs_names.equals(atac.obs_names):
            raise ValueError("RNA and ATAC cell orders differ")

        genes = pd.Index(pd.unique(pd.concat([triplets["tf"], triplets["gene"]])))
        peaks = pd.Index(triplets["peak"].unique())
        gene_pos = rna.var_names.get_indexer(genes)
        peak_pos = atac.var_names.get_indexer(peaks)
        if np.any(gene_pos < 0) or np.any(peak_pos < 0):
            raise ValueError("triplet features are absent from the h5mu matrices")

        # Read only triplet-relevant columns, preserving sparsity.
        rna_x = sparse.csc_matrix(rna.X[:, gene_pos])
        atac_x = sparse.csc_matrix(atac.X[:, peak_pos])
        gene_lookup = {name: i for i, name in enumerate(genes)}
        peak_lookup = {name: i for i, name in enumerate(peaks)}

        group_frame = rna.obs[GROUP_COLUMNS].astype(str)
        group_index = pd.MultiIndex.from_frame(group_frame)
        group_codes, groups = pd.factorize(group_index, sort=True)
        group_sizes = np.bincount(group_codes, minlength=len(groups)).astype(np.int64)
        counts = np.zeros((len(triplets), len(groups)), dtype=np.int64)

        for row_id, row in enumerate(triplets.itertuples(index=False)):
            tf_cells = _positive_rows(rna_x, gene_lookup[row.tf])
            gene_cells = _positive_rows(rna_x, gene_lookup[row.gene])
            peak_cells = _positive_rows(atac_x, peak_lookup[row.peak])
            active_cells = np.intersect1d(
                np.intersect1d(tf_cells, gene_cells, assume_unique=True),
                peak_cells,
                assume_unique=True,
            )
            if active_cells.size:
                counts[row_id] = np.bincount(
                    group_codes[active_cells], minlength=len(groups)
                )

        group_table = groups.to_frame(index=False)
        group_table.columns = GROUP_COLUMNS
        additions = []
        for group_id, group in group_table.iterrows():
            frame = triplets.copy()
            for column in GROUP_COLUMNS:
                frame[column] = group[column]
            frame["active_cell_count"] = counts[:, group_id]
            frame["group_total_cells"] = group_sizes[group_id]
            frame["active_cell_fraction"] = (
                frame["active_cell_count"] / frame["group_total_cells"]
            )
            additions.append(frame)
        additions = pd.concat(additions, ignore_index=True)
        result = scores.merge(
            additions,
            on=KEY_COLUMNS + GROUP_COLUMNS,
            how="left",
            validate="many_to_one",
        )
        if result[["active_cell_count", "group_total_cells", "active_cell_fraction"]].isna().any().any():
            raise ValueError("some active-score rows did not match an h5mu cell group")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        result.to_parquet(temporary, index=False)
        temporary.replace(output)
    finally:
        mdata.file.close()


def _valid_output(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        columns = pd.read_parquet(path).columns
    except Exception:
        return False
    return {
        "active_cell_count", "group_total_cells", "active_cell_fraction"
    }.issubset(columns)


def run_batch(tsv: Path, backend_root: Path, minimum_pairs: int) -> None:
    coverage = pd.read_csv(tsv, sep="\t")
    selected = coverage[
        pd.to_numeric(coverage["unique_tf_gene_pairs"], errors="coerce") >= minimum_pairs
    ]
    LOGGER.info("batch_start selected=%d minimum_pairs=%d", len(selected), minimum_pairs)
    successes = skipped = failures = 0
    for number, dataset in enumerate(selected["dataset"].astype(str), start=1):
        dataset_dir = backend_root / dataset
        h5mu = dataset_dir / f"{dataset}.h5mu"
        source = dataset_dir / "regulatory/triplet_active_scores.parquet"
        output = dataset_dir / "regulatory/triplet_active_scores.with_active_cells.parquet"
        if _valid_output(output):
            skipped += 1
            LOGGER.info("dataset_skip index=%d/%d dataset=%s output=%s", number, len(selected), dataset, output)
            continue
        try:
            LOGGER.info("dataset_start index=%d/%d dataset=%s", number, len(selected), dataset)
            add_active_cell_fraction(h5mu, source, output)
            successes += 1
            LOGGER.info("dataset_ok index=%d/%d dataset=%s output=%s", number, len(selected), dataset, output)
        except Exception:
            failures += 1
            LOGGER.exception("dataset_failed index=%d/%d dataset=%s", number, len(selected), dataset)
    LOGGER.info(
        "batch_complete selected=%d successes=%d skipped=%d failures=%d",
        len(selected), successes, skipped, failures,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5mu", type=Path)
    parser.add_argument("--active-scores", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--coverage-tsv", type=Path)
    parser.add_argument("--backend-root", type=Path)
    parser.add_argument("--minimum-pairs", type=int, default=100)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.coverage_tsv:
        if args.backend_root is None:
            parser.error("--backend-root is required with --coverage-tsv")
        run_batch(args.coverage_tsv, args.backend_root, args.minimum_pairs)
    else:
        if args.h5mu is None or args.active_scores is None or args.output is None:
            parser.error("--h5mu, --active-scores and --output are required in single-dataset mode")
        add_active_cell_fraction(args.h5mu, args.active_scores, args.output)


if __name__ == "__main__":
    main()
