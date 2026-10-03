from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import psutil
from scipy import sparse

from .logging_utils import get_logger


SOURCE_PRIORITY = {
    "ChIP-exo": 0,
    "ChIP-seq_and_ChIP-exo": 0,
    "ChIP-seq": 1,
    "GHT-SELEX": 2,
    "SELEX": 2,
    "SMiLE-seq": 2,
    "meSMiLE-seq": 2,
    "PBM": 3,
    "B1H": 3,
    "JASPAR": 4,
    "Transfac": 4,
    "Misc": 5,
}
GROUP_COLUMNS = ["project_id", "sample_id", "gpt_cell_type"]
LOGGER = get_logger("fimo")


def select_fimo_jobs(
    maximum_jobs: int,
    n_cells: int,
    available_memory_bytes: int | None = None,
) -> int:
    """Choose FIMO concurrency from dataset size and currently free memory."""
    if maximum_jobs < 1 or n_cells < 1:
        raise ValueError("maximum_jobs and n_cells must be positive")
    if n_cells <= 10_000:
        cell_cap = 48
    elif n_cells <= 25_000:
        cell_cap = 32
    elif n_cells <= 50_000:
        cell_cap = 24
    elif n_cells <= 100_000:
        cell_cap = 12
    else:
        cell_cap = 8
    if available_memory_bytes is None:
        available_memory_bytes = psutil.virtual_memory().available
    if available_memory_bytes < 1:
        raise ValueError("available_memory_bytes must be positive")
    # Use at most half of currently available RAM for FIMO workers, budgeting
    # 512 MiB per process. The other half remains for MuData and the OS.
    memory_cap = max(
        1,
        int(available_memory_bytes * 0.5 // (512 * 1024 * 1024)),
    )
    return max(1, min(maximum_jobs, cell_cap, memory_cap))


def _read_pwm(path: Path) -> np.ndarray:
    pwm = pd.read_csv(path, sep="\t").set_index("Pos")[["A", "C", "G", "T"]]
    values = pwm.to_numpy(dtype=float)
    if values.shape[0] < 4 or not np.isfinite(values).all():
        raise ValueError(f"invalid PWM: {path}")
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError(f"zero-sum PWM row: {path}")
    return values / totals


def _information_content(pwm: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(pwm > 0, pwm * np.log2(pwm / 0.25), 0.0)
    return float(terms.sum())


def _eligible_tf_expression(
    rna,
    candidate_tfs: pd.Index,
    min_cells: int,
    mean_threshold: float,
    max_threshold: float,
) -> pd.DataFrame:
    candidate_tfs = candidate_tfs[candidate_tfs.isin(rna.var_names)]
    if candidate_tfs.empty:
        raise ValueError("no CisDB TF names are present in RNA features")
    frame = rna.obs[GROUP_COLUMNS].astype(str)
    raw_codes, raw_groups = pd.factorize(pd.MultiIndex.from_frame(frame), sort=True)
    raw_sizes = np.bincount(raw_codes)
    keep_group = raw_sizes >= min_cells
    keep_cell = keep_group[raw_codes]
    remap = np.full(len(raw_groups), -1, dtype=int)
    remap[np.flatnonzero(keep_group)] = np.arange(keep_group.sum())
    codes = remap[raw_codes[keep_cell]]
    sizes = np.bincount(codes)
    if not len(sizes):
        raise ValueError("no pseudobulk groups pass min_cells for FIMO TF selection")
    indicator = sparse.csr_matrix(
        (
            (1.0 / sizes[codes]).astype(np.float32),
            (codes, np.arange(len(codes))),
        ),
        shape=(len(sizes), len(codes)),
    )
    positions = rna.var_names.get_indexer(candidate_tfs)
    values = sparse.csr_matrix(rna.X)[keep_cell][:, positions]
    pseudobulk = (indicator @ values).toarray()
    metrics = pd.DataFrame(
        {
            "TF_Name": candidate_tfs,
            "pseudobulk_expression_mean": pseudobulk.mean(axis=0),
            "pseudobulk_expression_max": pseudobulk.max(axis=0),
        }
    )
    metrics["passes_expression_filters"] = (
        (metrics["pseudobulk_expression_mean"] > mean_threshold)
        & (metrics["pseudobulk_expression_max"] > max_threshold)
    )
    return metrics


def build_cisdb_meme(
    cisdb_dir: Path,
    eligible_tfs: set[str],
    outdir: Path,
    max_motifs_per_tf: int = 1,
) -> tuple[Path, pd.DataFrame]:
    """Select representative CisDB PWMs and convert them to MEME format."""
    metadata_path = cisdb_dir / "TF_Information_all_motifs_plus.txt"
    pwm_dir = cisdb_dir / "pwms_all_motifs"
    if not metadata_path.is_file() or not pwm_dir.is_dir():
        raise FileNotFoundError(
            f"CisDB requires {metadata_path} and directory {pwm_dir}"
        )
    metadata = pd.read_csv(metadata_path, sep="\t", low_memory=False)
    required = {
        "TF_Name", "Motif_ID", "TF_Status", "MSource_Type",
        "MSource_Identifier",
    }
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"CisDB metadata missing columns: {sorted(missing)}")
    metadata["TF_Name"] = metadata["TF_Name"].astype(str)
    metadata = metadata[metadata["TF_Name"].isin(eligible_tfs)].copy()
    available = {path.stem: path for path in pwm_dir.glob("*.txt")}
    metadata = metadata[metadata["Motif_ID"].isin(available)].drop_duplicates(
        ["TF_Name", "Motif_ID", "TF_Status", "MSource_Type"]
    )
    pwm_cache, qc = {}, {}
    for motif_id in metadata["Motif_ID"].unique():
        try:
            pwm = _read_pwm(available[motif_id])
        except (ValueError, pd.errors.EmptyDataError):
            continue
        pwm_cache[motif_id] = pwm
        qc[motif_id] = (pwm.shape[0], _information_content(pwm))
    metadata = metadata[metadata["Motif_ID"].isin(pwm_cache)].copy()
    metadata["motif_length"] = metadata["Motif_ID"].map(
        {key: value[0] for key, value in qc.items()}
    )
    metadata["information_content"] = metadata["Motif_ID"].map(
        {key: value[1] for key, value in qc.items()}
    )
    metadata["source_priority"] = (
        metadata["MSource_Type"].map(SOURCE_PRIORITY).fillna(9)
    )
    selected = []
    for tf, group in metadata.groupby("TF_Name", sort=True):
        direct = group[group["TF_Status"].eq("D")]
        pool = direct if not direct.empty else group[group["TF_Status"].eq("I")]
        if pool.empty:
            pool = group
        chosen = (
            pool.sort_values(
                ["source_priority", "information_content", "Motif_ID"],
                ascending=[True, False, True],
            )
            .drop_duplicates("Motif_ID")
            .head(max_motifs_per_tf)
            .copy()
        )
        chosen["evidence_tier"] = "direct" if not direct.empty else "inferred"
        chosen["selected_rank"] = np.arange(1, len(chosen) + 1)
        selected.append(chosen)
    if not selected:
        raise ValueError("no valid CisDB PWM remains for eligible expressed TFs")
    catalog = pd.concat(selected, ignore_index=True)
    keep = [
        "TF_Name", "Motif_ID", "TF_Status", "evidence_tier",
        "selected_rank", "MSource_Type", "MSource_Identifier",
        "motif_length", "information_content",
    ]
    catalog = catalog[keep]
    catalog.to_csv(
        outdir / "representative_tf_pwm_catalog.tsv", sep="\t", index=False
    )
    LOGGER.info(
        "cisdb_catalog eligible_tfs=%d selected_tfs=%d motifs=%d",
        len(eligible_tfs), catalog["TF_Name"].nunique(),
        catalog["Motif_ID"].nunique(),
    )
    meme_path = outdir / "cisdb_representative_pwms.meme"
    with meme_path.open("w") as handle:
        handle.write(
            "MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
            "Background letter frequencies\n"
            "A 0.25 C 0.25 G 0.25 T 0.25\n\n"
        )
        for motif_id in catalog["Motif_ID"].drop_duplicates():
            pwm = pwm_cache[motif_id]
            handle.write(
                f"MOTIF {motif_id}\n"
                f"letter-probability matrix: alength= 4 w= {pwm.shape[0]} "
                "nsites= 20 E= 0\n"
            )
            np.savetxt(handle, pwm, fmt="%.8g")
            handle.write("\n")
    return meme_path, catalog


def _write_peak_fasta(
    peak_names: pd.Index,
    genome_fasta: Path,
    outdir: Path,
    bedtools_executable: str,
) -> tuple[Path, pd.DataFrame]:
    peaks = pd.DataFrame({"peak": peak_names.astype(str)})
    coords = peaks["peak"].str.extract(
        r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$"
    )
    if coords.isna().any(axis=None):
        bad = peaks.loc[coords.isna().any(axis=1), "peak"].head().tolist()
        raise ValueError(f"invalid ATAC peak names for FIMO: {bad}")
    coords[["start", "end"]] = coords[["start", "end"]].astype(np.int64)
    peaks = pd.concat([peaks, coords], axis=1)
    # Resolve chromosome aliases against the actual FASTA instead of assuming
    # one global naming convention. This matters for references such as
    # FlyBase/Ensembl dm6 where primary contigs may be a mixture of `X`, `2L`,
    # and a locally normalized `chr4`, while imported peaks use `chrX`/`chr4`.
    fai = genome_fasta.with_suffix(genome_fasta.suffix + ".fai")
    if fai.is_file():
        fasta_contigs = {
            line.split("\t", 1)[0] for line in fai.read_text().splitlines()
            if line.strip()
        }
    else:
        fasta_contigs = set()
        with genome_fasta.open() as handle:
            for line in handle:
                if line.startswith(">"):
                    fasta_contigs.add(line[1:].split()[0])

    def resolve_contig(chrom: str) -> str:
        if chrom in fasta_contigs:
            return chrom
        bare = chrom[3:] if chrom.lower().startswith("chr") else chrom
        candidates = [bare, f"chr{bare}"]
        matches = [candidate for candidate in candidates if candidate in fasta_contigs]
        if len(matches) == 1:
            return matches[0]
        return chrom

    peaks["input_chrom"] = peaks["chrom"]
    peaks["chrom"] = peaks["chrom"].map(resolve_contig)
    remapped = peaks[peaks["chrom"] != peaks["input_chrom"]]
    if not remapped.empty:
        mapping = (
            remapped[["input_chrom", "chrom"]].drop_duplicates()
            .sort_values(["input_chrom", "chrom"])
        )
        mapping.to_csv(outdir / "fimo_chromosome_aliases.tsv", sep="\t", index=False)
        LOGGER.info(
            "fimo_chromosome_aliases remapped_peaks=%d mappings=%s",
            len(remapped), mapping.to_dict("records"),
        )
    peaks["sequence_id"] = [f"peak_{i:09d}" for i in range(len(peaks))]
    bed = outdir / "atac_peaks.bed"
    raw_fasta = outdir / "atac_peaks.raw.fa"
    scan_fasta = outdir / "atac_peaks.fimo.fa"
    peaks[["chrom", "start", "end", "sequence_id"]].to_csv(
        bed, sep="\t", header=False, index=False
    )
    subprocess.run(
        [
            bedtools_executable, "getfasta", "-fi", str(genome_fasta),
            "-bed", str(bed), "-nameOnly", "-fo", str(raw_fasta),
        ],
        check=True,
    )
    expected = set(peaks["sequence_id"])
    observed = set()
    with raw_fasta.open() as source, scan_fasta.open("w") as target:
        for line in source:
            if line.startswith(">"):
                sequence_id = line[1:].split("::", 1)[0].split()[0]
                observed.add(sequence_id)
                target.write(f">{sequence_id}\n")
            else:
                target.write(line)
    raw_fasta.unlink()
    unexpected = observed - expected
    if unexpected:
        raise ValueError(
            f"bedtools returned unexpected sequence IDs: {sorted(unexpected)[:5]}"
        )
    missing = expected - observed
    if missing:
        dropped = peaks.loc[
            peaks["sequence_id"].isin(missing),
            ["sequence_id", "peak", "chrom", "start", "end"],
        ].copy()
        dropped.to_csv(
            outdir / "fimo_dropped_peaks.tsv", sep="\t", index=False
        )
        LOGGER.warning(
            "bedtools_missing_peaks dropped=%d extracted=%d requested=%d "
            "details=%s",
            len(dropped), len(observed), len(expected),
            outdir / "fimo_dropped_peaks.tsv",
        )
    mapping = peaks.loc[
        peaks["sequence_id"].isin(observed), ["sequence_id", "peak"]
    ].copy()
    if mapping.empty:
        raise ValueError("bedtools did not extract any ATAC peak sequences")
    mapping.to_csv(outdir / "fimo_sequence_id_map.tsv", sep="\t", index=False)
    return scan_fasta, mapping


def _split_fasta(path: Path, outdir: Path, sequences_per_chunk: int) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    chunks, handle, count = [], None, 0
    try:
        for line in path.open():
            if line.startswith(">") and count % sequences_per_chunk == 0:
                if handle is not None:
                    handle.close()
                chunk = outdir / f"chunk_{len(chunks):05d}.fa"
                chunks.append(chunk)
                handle = chunk.open("w")
            if handle is None:
                raise ValueError("FASTA does not begin with a header")
            handle.write(line)
            if line.startswith(">"):
                count += 1
    finally:
        if handle is not None:
            handle.close()
    if not chunks:
        raise ValueError("no sequences available for FIMO")
    return chunks


def _run_chunk(
    chunk: Path,
    meme: Path,
    result_root: Path,
    fimo_executable: str,
    threshold: float,
) -> Path:
    output = result_root / chunk.stem
    completed = subprocess.run(
        [
            fimo_executable, "--oc", str(output), "--thresh", str(threshold),
            "--verbosity", "1", str(meme), str(chunk),
        ],
        text=True,
        capture_output=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "process.log").write_text(
        f"STDOUT\n{completed.stdout}\nSTDERR\n{completed.stderr}\n"
    )
    if completed.returncode:
        raise RuntimeError(
            f"FIMO failed for {chunk.name} with exit code "
            f"{completed.returncode}: {completed.stderr[-2000:]}"
        )
    result = output / "fimo.tsv"
    if not result.exists():
        raise FileNotFoundError(result)
    return result


def run_fimo(
    dataset: str,
    peak_names: pd.Index,
    rna,
    genome_fasta: Path,
    cisdb_dir: Path,
    outdir: Path,
    threshold: float = 1e-5,
    jobs: int = 4,
    sequences_per_chunk: int = 2_000,
    fimo_executable: str = "fimo",
    bedtools_executable: str = "bedtools",
    min_pseudobulk_cells: int = 10,
    tf_pseudobulk_mean: float = 0.5,
    tf_pseudobulk_max: float = 0.5,
    max_motifs_per_tf: int = 1,
    forced_tfs: set[str] | None = None,
    target_valid_pwm_tfs: int = 3,
) -> Path:
    """Scan all retained ATAC peaks and return one best hit per TF/peak."""
    for executable in (fimo_executable, bedtools_executable):
        if shutil.which(executable) is None and not Path(executable).exists():
            raise FileNotFoundError(f"required executable not found: {executable}")
    for reference in (genome_fasta, cisdb_dir):
        if not reference.is_file():
            if not reference.is_dir():
                raise FileNotFoundError(reference)
    if jobs < 1 or sequences_per_chunk < 1:
        raise ValueError("FIMO jobs and sequences_per_chunk must be positive")
    outdir.mkdir(parents=True, exist_ok=True)
    cisdb_metadata = pd.read_csv(
        cisdb_dir / "TF_Information_all_motifs_plus.txt",
        sep="\t",
        usecols=["TF_Name"],
        dtype=str,
    )
    expression = _eligible_tf_expression(
        rna,
        pd.Index(cisdb_metadata["TF_Name"].dropna().unique()),
        min_pseudobulk_cells,
        tf_pseudobulk_mean,
        tf_pseudobulk_max,
    )
    forced_tfs = set() if forced_tfs is None else set(forced_tfs)
    absent_forced = forced_tfs - set(expression["TF_Name"])
    if absent_forced:
        LOGGER.warning("forced_tfs_absent_from_rna_or_cisdb tfs=%s", sorted(absent_forced))
    # Keep the configured thresholds as the default, but progressively relax
    # expression filtering when too few expressed TFs have a usable CisDB PWM.
    # This is dataset-adaptive without hard-coding dataset identifiers.
    threshold_ladder = [
        (tf_pseudobulk_mean, tf_pseudobulk_max),
        (0.25, 0.5),
        (0.25, 0.25),
        (0.1, 0.25),
    ]
    threshold_ladder = list(dict.fromkeys(
        (mean, maximum) for mean, maximum in threshold_ladder
        if mean <= tf_pseudobulk_mean and maximum <= tf_pseudobulk_max
    ))
    attempts = []
    selected = None
    last_valid = None
    if target_valid_pwm_tfs < 1:
        raise ValueError("target_valid_pwm_tfs must be positive")
    target_pwm_tfs = target_valid_pwm_tfs
    for mean_threshold, max_threshold in threshold_ladder:
        passes = (
            (expression["pseudobulk_expression_mean"] > mean_threshold)
            & (expression["pseudobulk_expression_max"] > max_threshold)
        )
        passes |= expression["TF_Name"].isin(forced_tfs)
        eligible_tfs = set(expression.loc[passes, "TF_Name"])
        try:
            candidate_meme, candidate_catalog = build_cisdb_meme(
                cisdb_dir, eligible_tfs, outdir, max_motifs_per_tf
            )
            valid_pwm_tfs = int(candidate_catalog["TF_Name"].nunique())
            last_valid = (
                mean_threshold, max_threshold, passes,
                candidate_meme, candidate_catalog,
            )
        except ValueError as exc:
            if str(exc) != "no valid CisDB PWM remains for eligible expressed TFs":
                raise
            valid_pwm_tfs = 0
        attempts.append({
            "tf_pseudobulk_mean": mean_threshold,
            "tf_pseudobulk_max": max_threshold,
            "eligible_expression_tfs": int(passes.sum()),
            "valid_pwm_tfs": valid_pwm_tfs,
        })
        LOGGER.info(
            "cisdb_threshold_attempt mean=%g max=%g eligible_tfs=%d "
            "valid_pwm_tfs=%d target=%d",
            mean_threshold, max_threshold, int(passes.sum()),
            valid_pwm_tfs, target_pwm_tfs,
        )
        if valid_pwm_tfs >= target_pwm_tfs:
            selected = last_valid
            break
    if selected is None:
        selected = last_valid
    if selected is None:
        raise ValueError(
            "no valid CisDB PWM remains after adaptive TF expression fallback"
        )
    (
        selected_mean, selected_max, selected_passes, meme_file, catalog,
    ) = selected
    expression["passes_expression_filters"] = selected_passes
    expression["selected_tf_pseudobulk_mean"] = selected_mean
    expression["selected_tf_pseudobulk_max"] = selected_max
    expression.to_csv(outdir / "tf_expression_metrics.csv", index=False)
    (outdir / "tf_expression_threshold_selection.json").write_text(
        json.dumps({
            "configured": {
                "tf_pseudobulk_mean": tf_pseudobulk_mean,
                "tf_pseudobulk_max": tf_pseudobulk_max,
            },
            "target_valid_pwm_tfs": target_pwm_tfs,
            "forced_tfs": sorted(forced_tfs),
            "attempts": attempts,
            "selected": {
                "tf_pseudobulk_mean": selected_mean,
                "tf_pseudobulk_max": selected_max,
                "eligible_expression_tfs": int(selected_passes.sum()),
                "valid_pwm_tfs": int(catalog["TF_Name"].nunique()),
            },
        }, indent=2) + "\n"
    )
    scan_fasta, sequence_map = _write_peak_fasta(
        peak_names, genome_fasta, outdir, bedtools_executable
    )
    chunks = _split_fasta(
        scan_fasta, outdir / "fasta_chunks", sequences_per_chunk
    )
    result_root = outdir / "fimo_chunks"
    result_root.mkdir(exist_ok=True)
    LOGGER.info(
        "fimo_scan_start peaks=%d chunks=%d jobs=%d threshold=%g",
        len(peak_names), len(chunks), min(jobs, len(chunks)), threshold,
    )
    with ThreadPoolExecutor(max_workers=min(jobs, len(chunks))) as pool:
        futures = [
            pool.submit(
                _run_chunk, chunk, meme_file, result_root, fimo_executable,
                threshold,
            )
            for chunk in chunks
        ]
        result_paths = [future.result() for future in futures]

    frames = []
    retained_raw_hits = 0
    sequence_lookup = sequence_map.set_index("sequence_id")["peak"].to_dict()
    motif_to_tf = catalog[["Motif_ID", "TF_Name"]].drop_duplicates()
    for path in result_paths:
        try:
            frame = pd.read_csv(path, sep="\t", comment="#")
        except pd.errors.EmptyDataError:
            # FIMO writes a valid, comment-only fimo.tsv when a FASTA chunk
            # has no motif hit passing the requested threshold.
            LOGGER.info("fimo_empty_chunk path=%s", path)
            continue
        if frame.empty:
            continue
        frame = frame.rename(
            columns={
                "sequence_name": "sequence_id",
                "p-value": "motif_pvalue",
                "q-value": "motif_qvalue",
                "motif_id": "Motif_ID",
                "motif_alt_id": "Motif_Alt_ID",
            }
        )
        required = {"sequence_id", "Motif_ID", "motif_pvalue"}
        missing = required - set(frame)
        if missing:
            raise ValueError(f"FIMO output missing columns: {sorted(missing)}")
        frame = frame[frame["motif_pvalue"] < threshold].copy()
        if frame.empty:
            continue
        frame["peak"] = frame["sequence_id"].map(sequence_lookup)
        if frame["peak"].isna().any():
            raise ValueError("FIMO returned unknown sequence IDs")
        frame = frame.merge(
            motif_to_tf, on="Motif_ID", how="inner", validate="many_to_many"
        )
        retained_raw_hits += len(frame)
        if frame.empty:
            continue
        # Every peak belongs to exactly one FASTA chunk, so reducing each
        # chunk before concatenation is equivalent to a global TF/peak reduce.
        frame = (
            frame.sort_values("motif_pvalue")
            .drop_duplicates(["TF_Name", "peak"], keep="first")
        )
        frames.append(frame)
    if not frames:
        raise ValueError(f"{dataset}: FIMO produced no motif hits")
    tf_peak = pd.concat(frames, ignore_index=True)
    tf_peak.insert(0, "dataset", dataset)
    output = outdir / "tf_peak_hits.parquet"
    building = output.with_name(f".{output.stem}.building{output.suffix}")
    tf_peak.to_parquet(building, index=False)
    building.replace(output)
    summary = {
        "dataset": dataset,
        "retained_atac_peaks": int(len(peak_names)),
        "fasta_chunks": len(chunks),
        "fimo_jobs": jobs,
        "fimo_pvalue_threshold": threshold,
        "raw_motif_hits": int(retained_raw_hits),
        "unique_tf_peak_hits": int(len(tf_peak)),
        "unique_tfs": int(tf_peak["TF_Name"].nunique()),
        "eligible_expressed_tfs": len(eligible_tfs),
        "selected_cisdb_tfs": int(catalog["TF_Name"].nunique()),
        "selected_cisdb_motifs": int(catalog["Motif_ID"].nunique()),
        "genome_fasta": str(genome_fasta),
        "cisdb_dir": str(cisdb_dir),
        "generated_meme_file": str(meme_file),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    LOGGER.info(
        "fimo_scan_complete raw_hits=%d unique_tf_peak_hits=%d unique_tfs=%d output=%s",
        retained_raw_hits, len(tf_peak), tf_peak["TF_Name"].nunique(), output,
    )
    return output
