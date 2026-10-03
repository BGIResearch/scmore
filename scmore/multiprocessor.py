#!/usr/bin/env python
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import mudata as md
import pandas as pd

if __package__:
    from .annotation import annotate_mdata, build_annotation_payload, request_annotations, validate_annotations
    from .config import PipelineConfig
    from .context_means import build_context_means
    from .embeddings import add_umaps, strip_wnn
    from .export import export_backend
    from .fimo import run_fimo, select_fimo_jobs
    from .gene_activity import build_gene_activity
    from .metadata import add_metadata, dataset_organism, dataset_source_and_organism, read_dataset_metadata
    from .logging_utils import configure_logging, logged_stage, run_stage
    from .preprocess import atomic_write_h5mu, cluster_and_markers, load_filter_annotate
    from .regulatory import build_triplets, compute_active_scores, compute_peak_gene_links
    from .run_rapids_h5mu_compat import run_one as run_rapids_embeddings
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scmore.annotation import annotate_mdata, build_annotation_payload, request_annotations, validate_annotations
    from scmore.config import PipelineConfig
    from scmore.context_means import build_context_means
    from scmore.embeddings import add_umaps, strip_wnn
    from scmore.export import export_backend
    from scmore.fimo import run_fimo, select_fimo_jobs
    from scmore.gene_activity import build_gene_activity
    from scmore.metadata import add_metadata, dataset_organism, dataset_source_and_organism, read_dataset_metadata
    from scmore.logging_utils import configure_logging, logged_stage, run_stage
    from scmore.preprocess import atomic_write_h5mu, cluster_and_markers, load_filter_annotate
    from scmore.regulatory import build_triplets, compute_active_scores, compute_peak_gene_links
    from scmore.run_rapids_h5mu_compat import run_one as run_rapids_embeddings


def _checkpoint(path: Path, resume: bool) -> bool:
    return resume and path.exists()


def _complete_outputs(final_dir: Path, dataset: str) -> list[Path]:
    return [
        final_dir / f"{dataset}.h5mu",
        final_dir / f"{dataset}.cell_annotations.json",
        final_dir / f"{dataset}.cluster_annotations.csv",
        final_dir / "meta/cells.parquet",
        final_dir / "embedding/umap_rna.parquet",
        final_dir / "embedding/umap_atac.parquet",
        final_dir / "expression/rna_expr.parquet",
        final_dir / "expression/atac_expr.parquet",
        final_dir / "expression/gene_activity.parquet",
        final_dir / "expression/gene_mean_context.parquet",
        final_dir / "expression/peak_mean_context.parquet",
        final_dir / "regulatory/peak_gene_links.parquet",
        final_dir / "regulatory/tf_peak_motif_hits.parquet",
        final_dir / "regulatory/tf_peak_gene_triplets.parquet",
        final_dir / "regulatory/triplet_active_scores.parquet",
    ]


def _latest_log_completed(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    latest_complete = -1
    latest_failed = -1
    with log_path.open(errors="replace") as handle:
        for position, line in enumerate(handle):
            if "dataset_complete " in line:
                latest_complete = position
            elif "stage_failed " in line:
                latest_failed = position
    return latest_complete > latest_failed


def _process_dataset_unlocked(
    config_dict: dict, dataset: str, worker_id: int = 0
) -> dict:
    config = PipelineConfig(**{
        **config_dict,
        "thresholds": __import__("scmore.config", fromlist=["Thresholds"]).Thresholds(**config_dict["thresholds"]),
    })
    work = config.work_root / dataset
    final_dir = config.output_root / dataset
    source_dataset, _ = dataset_source_and_organism(dataset)
    input_dir = config.data_root / source_dataset
    work.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(
        work / "pipeline.log", dataset, config.log_level, console=True
    )
    logger.info(
        "dataset_start worker_id=%d resume=%s input=%s output=%s",
        worker_id, config.resume, input_dir, final_dir,
    )
    h5mu = final_dir / f"{dataset}.h5mu"
    markers_path = work / f"{dataset}.markers.csv"
    mapping_path = final_dir / f"{dataset}.cluster_annotations.csv"
    annotation_json = final_dir / f"{dataset}.cell_annotations.json"
    links_path = work / "peak_gene_links.parquet"
    triplets_path = work / "tf_peak_gene_triplets.parquet"
    active_path = final_dir / "regulatory/triplet_active_scores.parquet"

    required_outputs = _complete_outputs(final_dir, dataset)
    if (
        config.resume
        and all(path.is_file() for path in required_outputs)
        and _latest_log_completed(work / "pipeline.log")
    ):
        logger.info(
            "dataset_skipped_complete outputs=%d h5mu=%s",
            len(required_outputs), h5mu,
        )
        return {
            "dataset": dataset, "status": "skipped", "h5mu": str(h5mu)
        }

    metadata = run_stage(
        logger, "metadata", read_dataset_metadata, config.metadata, dataset,
        config.sample_file_map,
    )
    organism = dataset_organism(metadata)
    logger.info("metadata_loaded organism=%s samples=%d", organism, len(metadata))
    if organism not in config.gtf_by_organism:
        raise ValueError(f"{dataset}: unsupported organism {organism}")

    if _checkpoint(h5mu, config.resume):
        mdata = run_stage(logger, "load_h5mu_checkpoint", md.read_h5mu, h5mu)
        logger.info("checkpoint_reused path=%s", h5mu)
    else:
        mdata = run_stage(
            logger,
            "read_filter_peak_annotation",
            load_filter_annotate,
            input_dir,
            metadata["sample_name"].astype(str).tolist(),
            metadata.get(
                "sample_file", metadata["sample_name"]
            ).astype(str).tolist(),
            config.gtf(dataset, organism),
            config.thresholds.feature_fraction,
        )
        logger.info(
            "matrix_ready cells=%d genes=%d peaks=%d",
            mdata.mod["rna"].n_obs, mdata.mod["rna"].n_vars,
            mdata.mod["atac"].n_vars,
        )
        run_stage(logger, "add_metadata", add_metadata, mdata, metadata)
        run_stage(
            logger, "cluster_markers", cluster_and_markers, mdata, markers_path
        )
        run_stage(logger, "write_h5mu", atomic_write_h5mu, h5mu, mdata)

    if "project_id" not in mdata.mod["rna"].obs:
        run_stage(logger, "add_metadata_resume", add_metadata, mdata, metadata)
    removed_wnn = strip_wnn(mdata)
    if "leiden" not in mdata.mod["rna"].obs or not markers_path.exists():
        run_stage(
            logger, "cluster_markers_resume", cluster_and_markers,
            mdata, markers_path,
        )

    annotation_added = "gpt_cell_type" not in mdata.mod["rna"].obs
    if annotation_added:
        with logged_stage(logger, "gpt_cell_annotation"):
            markers = pd.read_csv(markers_path)
            payload = build_annotation_payload(
                dataset, markers, metadata, config.top_markers
            )
            clusters = set(mdata.mod["rna"].obs["leiden"].astype(str))
            for attempt in range(1, 4):
                response = request_annotations(
                    payload, config.annotation_model, config.annotation_response,
                    config.annotation_base_url, config.annotation_api_key_env,
                    config.annotation_mode, config.codex_executable, work,
                )
                try:
                    mapping = validate_annotations(response, clusters)
                    break
                except ValueError:
                    if attempt == 3 or config.annotation_response:
                        raise
                    logger.warning(
                        "annotation_validation_retry attempt=%d max_attempts=3",
                        attempt,
                        exc_info=True,
                    )
            inconsistent = mapping.loc[
                ~mapping["tissue_consistent"], "old_cluster"
            ].tolist()
            if inconsistent:
                logger.warning(
                    "annotation_tissue_inconsistent clusters=%s; retaining "
                    "annotations with tissue_consistent=false",
                    inconsistent,
                )
            annotate_mdata(mdata, mapping, annotation_json, mapping_path)
            logger.info(
                "annotation_complete clusters=%d cell_types=%d",
                len(mapping), mapping["cell_type"].nunique(),
            )

    embedded = (
        "X_umap" in mdata.mod["rna"].obsm
        and "X_umap" in mdata.mod["atac"].obsm
    )
    if not (config.resume and embedded):
        with logged_stage(logger, "rna_atac_embeddings"):
            if config.run_rapids:
                gpu = config.gpu_ids[worker_id % len(config.gpu_ids)]
                logger.info("embedding_backend=rapids gpu=%s", gpu)
                atomic_write_h5mu(h5mu, mdata)
                try:
                    run_rapids_embeddings(
                        h5mu, gpu=gpu, rapids_env=config.rapids_env,
                        random_state=config.random_seed,
                    )
                    mdata = md.read_h5mu(h5mu)
                except subprocess.CalledProcessError:
                    logger.warning(
                        "rapids_embedding_failed; falling back to CPU sparse "
                        "PCA/neighbors/UMAP",
                        exc_info=True,
                    )
                    add_umaps(mdata, seed=config.random_seed)
                    atomic_write_h5mu(h5mu, mdata)
            else:
                logger.info("embedding_backend=cpu")
                add_umaps(mdata, seed=config.random_seed)
                atomic_write_h5mu(h5mu, mdata)
    elif annotation_added or removed_wnn:
        run_stage(logger, "write_h5mu_updates", atomic_write_h5mu, h5mu, mdata)

    if _checkpoint(links_path, config.resume):
        links = run_stage(
            logger, "load_peak_gene_links_checkpoint",
            pd.read_parquet, links_path,
        )
    else:
        links = run_stage(
            logger, "peak_gene_links", compute_peak_gene_links,
            mdata, links_path, config.thresholds.min_pseudobulk_cells,
        )
    logger.info("peak_gene_links rows=%d", len(links))
    configured_motif_path = config.motif_hits.get(dataset)
    generated_motif_path = work / "fimo/tf_peak_hits.parquet"
    if configured_motif_path is not None:
        motif_path = configured_motif_path
        if not motif_path.is_file():
            raise FileNotFoundError(motif_path)
        logger.info("fimo_external_checkpoint path=%s", motif_path)
    elif _checkpoint(generated_motif_path, config.resume):
        motif_path = generated_motif_path
        logger.info("fimo_checkpoint_reused path=%s", motif_path)
    else:
        missing_references = [
            name for name, mapping in (
                ("genome FASTA", config.genome_fasta_by_organism),
                ("CisDB", config.cisdb_by_organism),
            )
            if organism not in mapping
        ]
        if missing_references:
            raise ValueError(
                f"{dataset}: FIMO references are not configured for {organism}: "
                f"{missing_references}"
            )
        effective_fimo_jobs = select_fimo_jobs(
            config.fimo_jobs, mdata.mod["rna"].n_obs
        )
        logger.info(
            "fimo_concurrency selected=%d configured_max=%d cells=%d peaks=%d",
            effective_fimo_jobs, config.fimo_jobs,
            mdata.mod["rna"].n_obs, mdata.mod["atac"].n_vars,
        )
        motif_path = run_stage(
            logger,
            "fimo",
            run_fimo,
            dataset,
            mdata.mod["atac"].var_names,
            mdata.mod["rna"],
            config.genome_fasta(dataset, organism),
            config.cisdb_by_organism[organism],
            generated_motif_path.parent,
            threshold=config.thresholds.motif_pvalue,
            jobs=effective_fimo_jobs,
            sequences_per_chunk=config.fimo_sequences_per_chunk,
            fimo_executable=config.fimo_executable,
            bedtools_executable=config.bedtools_executable,
            min_pseudobulk_cells=config.thresholds.min_pseudobulk_cells,
            tf_pseudobulk_mean=config.thresholds.tf_pseudobulk_mean,
            tf_pseudobulk_max=config.thresholds.tf_pseudobulk_max,
        )
    motifs = run_stage(logger, "load_motif_hits", pd.read_parquet, motif_path)
    logger.info("motif_hits rows=%d", len(motifs))
    if _checkpoint(triplets_path, config.resume):
        triplets = run_stage(
            logger, "load_triplets_checkpoint", pd.read_parquet, triplets_path
        )
    else:
        triplets = run_stage(
            logger, "triplets", build_triplets,
            mdata, links, motifs, triplets_path, config.thresholds,
        )
    logger.info("triplets rows=%d", len(triplets))
    if not _checkpoint(active_path, config.resume):
        run_stage(
            logger, "triplet_active_scores", compute_active_scores,
            dataset, mdata, triplets, active_path,
            config.thresholds.min_pseudobulk_cells,
        )
    else:
        logger.info("active_score_checkpoint_reused path=%s", active_path)
    run_stage(
        logger, "context_means", build_context_means,
        mdata, final_dir,
    )
    run_stage(
        logger, "export_backend", export_backend,
        dataset, mdata, final_dir, config.export_max_cells,
        config.export_min_per_stratum, config.random_seed,
        links, motifs, triplets,
    )
    run_stage(
        logger, "gene_activity", build_gene_activity,
        final_dir, True,
    )
    logger.info(
        "dataset_complete cells=%d genes=%d peaks=%d triplets=%d h5mu=%s",
        mdata.mod["rna"].n_obs, mdata.mod["rna"].n_vars,
        mdata.mod["atac"].n_vars, len(triplets), h5mu,
    )
    return {
        "dataset": dataset, "status": "ok", "h5mu": str(h5mu),
        "cells": mdata.mod["rna"].n_obs, "triplets": len(triplets),
    }


def process_dataset(config_dict: dict, dataset: str, worker_id: int = 0) -> dict:
    """Serialize all writes belonging to the same logical dataset."""
    work_root = Path(config_dict["work_root"])
    lock_dir = work_root / ".dataset_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{dataset}.lock"
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.write(f"pid={os.getpid()}\n")
        lock_handle.flush()
        try:
            return _process_dataset_unlocked(config_dict, dataset, worker_id)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel scMORE v2 dataset processor")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--datasets", nargs="*", help="override config datasets")
    parser.add_argument("--workers", type=int, help="override worker count")
    parser.add_argument(
        "--fimo-jobs", type=int,
        help="override maximum FIMO concurrency for this batch",
    )
    args = parser.parse_args()
    config = PipelineConfig.from_json(args.config)
    if args.datasets:
        config.datasets = args.datasets
    if args.workers:
        config.workers = args.workers
    if args.fimo_jobs:
        if args.fimo_jobs < 1:
            parser.error("--fimo-jobs must be >= 1")
        config.fimo_jobs = args.fimo_jobs
    config.work_root.mkdir(parents=True, exist_ok=True)
    config.output_root.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(
        config.work_root / "multiprocessor.log",
        dataset="-",
        level=config.log_level,
        console=True,
    )
    logger.info(
        "multiprocessor_start datasets=%s workers=%d",
        ",".join(config.datasets), config.workers,
    )
    run_id = f"{datetime.now().strftime('%Y%m%dT%H%M%S')}.{os.getpid()}"
    config_dict = config.to_dict()
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        futures = {
            pool.submit(process_dataset, config_dict, dataset, i): dataset
            for i, dataset in enumerate(config.datasets)
        }
        for future in as_completed(futures):
            dataset = futures[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(
                    "dataset_result %s",
                    json.dumps(result, ensure_ascii=False),
                )
            except Exception as exc:
                failure = {"dataset": dataset, "status": "failed", "error": str(exc)}
                failures.append(failure)
                logger.exception(
                    "dataset_failed dataset=%s error=%s", dataset, exc
                )
    report = {"completed": results, "failed": failures}
    report_text = json.dumps(report, ensure_ascii=False, indent=2)
    report_path = config.work_root / f"multiprocessor_report.{run_id}.json"
    report_path.write_text(report_text)
    latest_path = config.work_root / "multiprocessor_report.json"
    latest_building = config.work_root / f".multiprocessor_report.{run_id}.building"
    latest_building.write_text(report_text)
    latest_building.replace(latest_path)
    logger.info(
        "multiprocessor_complete succeeded=%d failed=%d report=%s",
        len(results), len(failures), report_path,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
