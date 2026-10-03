from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import duckdb
import pandas as pd


LOGGER = logging.getLogger(__name__)
SOURCE_NAME = "triplet_active_scores.with_active_cells.parquet"
DEFAULT_OUTPUT_NAME = "triplet_active_scores.dataset_active_gt10pct_n20.parquet"


def filter_one(source: Path, output: Path, score_threshold: float, count_threshold: int) -> dict:
    con = duckdb.connect()
    source_sql = str(source).replace("'", "''")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary_sql = str(temporary).replace("'", "''")
    query = f"""
    WITH aggregated AS (
      SELECT dataset, tf, peak, gene, gpt_cell_type,
             sum(CAST(active_cell_count AS BIGINT)) AS dataset_active_cell_count,
             sum(CAST(group_total_cells AS BIGINT)) AS dataset_group_total_cells,
             sum(CAST(active_cell_count AS DOUBLE))
               / nullif(sum(CAST(group_total_cells AS DOUBLE)), 0)
               AS dataset_active_cell_fraction
      FROM read_parquet('{source_sql}')
      GROUP BY dataset, tf, peak, gene, gpt_cell_type
      HAVING dataset_active_cell_fraction > {score_threshold!r}
         AND dataset_active_cell_count > {count_threshold:d}
    )
    SELECT s.*,
           a.dataset_active_cell_count,
           a.dataset_group_total_cells,
           a.dataset_active_cell_fraction
    FROM read_parquet('{source_sql}') AS s
    INNER JOIN aggregated AS a
      USING (dataset, tf, peak, gene, gpt_cell_type)
    """
    try:
        source_rows = con.execute(
            f"SELECT count(*) FROM read_parquet('{source_sql}')"
        ).fetchone()[0]
        con.execute(
            f"COPY ({query}) TO '{temporary_sql}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        kept_rows, kept_relations = con.execute(f"""
            SELECT count(*), count(DISTINCT concat_ws(
                '\x1f', dataset, tf, peak, gene, gpt_cell_type
            ))
            FROM read_parquet('{temporary_sql}')
        """).fetchone()
        temporary.replace(output)
        return {
            "dataset": source.parent.parent.name,
            "status": "ok",
            "source_rows": int(source_rows),
            "kept_rows": int(kept_rows),
            "kept_relations": int(kept_relations),
            "retained_fraction": float(kept_rows / source_rows) if source_rows else 0.0,
            "output": str(output),
        }
    finally:
        con.close()


def run_batch(
    backend_root: Path, summary: Path, score_threshold: float,
    count_threshold: int, output_name: str,
) -> None:
    sources = sorted(backend_root.glob(f"*/regulatory/{SOURCE_NAME}"))
    records = []
    LOGGER.info("batch_start datasets=%d", len(sources))
    for index, source in enumerate(sources, start=1):
        output = source.with_name(output_name)
        if output.exists():
            record = {
                "dataset": source.parent.parent.name,
                "status": "skipped_existing",
                "source_rows": None,
                "kept_rows": None,
                "kept_relations": None,
                "retained_fraction": None,
                "output": str(output),
            }
            LOGGER.info("dataset_skip index=%d/%d dataset=%s", index, len(sources), record["dataset"])
        else:
            try:
                record = filter_one(source, output, score_threshold, count_threshold)
                LOGGER.info(
                    "dataset_ok index=%d/%d dataset=%s kept_rows=%d kept_relations=%d",
                    index, len(sources), record["dataset"], record["kept_rows"], record["kept_relations"],
                )
            except Exception as exc:
                record = {
                    "dataset": source.parent.parent.name,
                    "status": "failed",
                    "source_rows": None,
                    "kept_rows": None,
                    "kept_relations": None,
                    "retained_fraction": None,
                    "output": str(output),
                    "error": repr(exc),
                }
                LOGGER.exception("dataset_failed index=%d/%d dataset=%s", index, len(sources), record["dataset"])
        records.append(record)
        pd.DataFrame(records).to_csv(summary, sep="\t", index=False)
    LOGGER.info(
        "batch_complete ok=%d skipped=%d failed=%d",
        sum(r["status"] == "ok" for r in records),
        sum(r["status"] == "skipped_existing" for r in records),
        sum(r["status"] == "failed" for r in records),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.10)
    parser.add_argument("--count-threshold", type=int, default=20)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    if Path(args.output_name).name != args.output_name or not args.output_name.endswith(".parquet"):
        parser.error("--output-name must be a plain .parquet filename")
    run_batch(
        args.backend_root, args.summary, args.score_threshold,
        args.count_threshold, args.output_name,
    )


if __name__ == "__main__":
    main()
