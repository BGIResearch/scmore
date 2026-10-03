from __future__ import annotations

import os
from pathlib import Path

import duckdb
import psutil

from .logging_utils import get_logger


LOGGER = get_logger("gene_activity")


def _sql_path(path: Path) -> str:
    """Return a DuckDB-safe absolute path literal."""
    return path.resolve().as_posix().replace("'", "''")


def build_gene_activity(dataset_dir: Path, overwrite: bool = False) -> Path:
    """Build a long-form gene-by-cell activity table from exported ATAC data."""
    dataset_dir = dataset_dir.resolve()
    atac_path = dataset_dir / "expression/atac_expr.parquet"
    links_path = dataset_dir / "regulatory/peak_gene_links.parquet"
    output_path = dataset_dir / "expression/gene_activity.parquet"
    temporary_path = output_path.with_suffix(".parquet.tmp")
    duckdb_temp = dataset_dir / "expression/.duckdb_gene_activity_tmp"

    for input_path in (atac_path, links_path):
        if not input_path.is_file():
            raise FileNotFoundError(f"missing gene-activity input: {input_path}")
    if output_path.exists() and not overwrite:
        LOGGER.info("gene_activity_checkpoint_reused path=%s", output_path)
        return output_path
    if temporary_path.exists():
        temporary_path.unlink()

    duckdb_temp.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        memory_mib = max(
            512,
            int(psutil.virtual_memory().available / (1024 * 1024) * 0.5),
        )
        connection.execute(f"SET memory_limit='{memory_mib}MiB'")
        connection.execute(f"SET threads={min(4, os.cpu_count() or 1)}")
        connection.execute(
            f"SET temp_directory='{_sql_path(duckdb_temp)}'"
        )
        connection.execute(
            f"""
            COPY (
                SELECT
                    links.gene,
                    atac.cell_id,
                    SUM(atac.value) AS activity
                FROM read_parquet('{_sql_path(atac_path)}') AS atac
                INNER JOIN read_parquet('{_sql_path(links_path)}') AS links
                    ON atac.peak = links.peak
                WHERE links.gene IS NOT NULL
                GROUP BY links.gene, atac.cell_id
            )
            TO '{_sql_path(temporary_path)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        temporary_path.replace(output_path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    finally:
        connection.close()

    LOGGER.info("gene_activity_complete output=%s", output_path)
    return output_path
