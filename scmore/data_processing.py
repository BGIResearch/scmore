from __future__ import annotations

import gc
import gzip
from pathlib import Path

import anndata as ad
import mudata as md
import muon as mu
import numpy as np
import pandas as pd
import pyranges as pr
import scanpy as sc
from scipy import sparse

from .logging_utils import get_logger


GTF_BY_ORGANISM = {
    "Homo sapiens": Path("/path/to/reference_data/human/genes.gtf.gz"),
    "Mus musculus": Path("/path/to/reference_data/mouse/genes.gtf.gz"),
    "Danio rerio": Path("/path/to/reference_data/zebrafish/genes.gtf.gz"),
    "Gallus gallus": Path("/path/to/reference_data/chicken/genes.gtf.gz"),
    "Canis lupus familiaris": Path("/path/to/reference_data/dog/genes.gtf.gz"),
    "Rattus norvegicus": Path("/path/to/reference_data/rat/genes.gtf.gz"),
    "Drosophila melanogaster": Path("/path/to/reference_data/fly/genes.gtf.gz"),
    "Macaca mulatta": Path("/path/to/reference_data/macaque/genes.gtf.gz"),
}
LOGGER = get_logger("data")
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def _is_10x_dir(path: Path) -> bool:
    names = [p.name for p in path.iterdir() if p.is_file()]
    return (
        any(x.startswith("matrix") for x in names)
        and any(x.startswith("features") for x in names)
        and any(x.startswith("barcodes") for x in names)
    )


def _is_hdf5(path: Path) -> bool:
    """Return whether a file starts with the HDF5 on-disk signature."""
    try:
        with path.open("rb") as handle:
            return handle.read(len(HDF5_SIGNATURE)) == HDF5_SIGNATURE
    except OSError:
        return False


def _clean_modalities(mdata):
    if not {"rna", "atac"} <= set(mdata.mod):
        raise ValueError(f"expected rna and atac modalities; observed {list(mdata.mod)}")
    rna = mdata.mod["rna"]
    rna.var_names = rna.var_names.astype(str)
    rna._inplace_subset_var(~rna.var_names.duplicated())
    rna.var_names_make_unique()
    atac = mdata.mod["atac"]
    atac.var_names = atac.var_names.astype(str)
    atac._inplace_subset_var(~atac.var_names.duplicated())
    common = rna.obs_names.intersection(atac.obs_names, sort=False)
    if common.empty:
        raise ValueError("RNA and ATAC have no common cells")
    rna._inplace_subset_obs(rna.obs_names.isin(common))
    atac._inplace_subset_obs(atac.obs_names.isin(common))
    if not rna.obs_names.equals(atac.obs_names):
        atac = atac[rna.obs_names].copy()
    # Rebuild only the lightweight MuData container. Reusing the AnnData
    # objects avoids matrix copies and discards stale top-level duplicate var
    # indices that cannot be updated safely.
    cleaned = md.MuData({"rna": rna, "atac": atac})
    cleaned.uns = mdata.uns
    return cleaned


def read_single(path: Path):
    if path.is_file():
        if path.suffix == ".h5mu":
            return _clean_modalities(md.read_h5mu(path))
        if path.suffix == ".h5":
            if not _is_hdf5(path):
                raise ValueError(f"file has .h5 suffix but is not HDF5: {path}")
            return _clean_modalities(mu.read_10x_h5(path))
        raise ValueError(f"unsupported input file: {path}")
    h5mu = sorted(path.glob("*.h5mu"))
    h5 = sorted(path.glob("*.h5"))
    if h5mu:
        return _clean_modalities(md.read_h5mu(h5mu[0]))
    if h5:
        valid_h5 = [candidate for candidate in h5 if _is_hdf5(candidate)]
        if valid_h5:
            return _clean_modalities(mu.read_10x_h5(valid_h5[0]))
        if _is_10x_dir(path):
            LOGGER.warning(
                "invalid_h5_fallback_to_mtx path=%s invalid_h5=%s",
                path, [candidate.name for candidate in h5],
            )
            return _clean_modalities(mu.read_10x_mtx(path))
        raise ValueError(
            f".h5 file is not HDF5 and no complete 10x MTX files exist in {path}: "
            f"{[candidate.name for candidate in h5]}"
        )
    if _is_10x_dir(path):
        return _clean_modalities(mu.read_10x_mtx(path))
    raise ValueError(f"no h5mu, 10x h5, or 10x mtx data in {path}")


def _qc(mdata, min_genes: int, min_cells: int, min_atac_counts: int) -> None:
    rna, atac = mdata.mod["rna"], mdata.mod["atac"]
    sc.pp.calculate_qc_metrics(rna, inplace=True)
    sc.pp.calculate_qc_metrics(atac, inplace=True)
    rna._inplace_subset_var(rna.var["n_cells_by_counts"].to_numpy() >= min_cells)
    rna._inplace_subset_obs(rna.obs["n_genes_by_counts"].to_numpy() >= min_genes)
    atac._inplace_subset_var(atac.var["n_cells_by_counts"].to_numpy() >= min_cells)
    atac._inplace_subset_obs(atac.obs["total_counts"].to_numpy() >= min_atac_counts)
    common = rna.obs_names.intersection(atac.obs_names, sort=False)
    rna._inplace_subset_obs(rna.obs_names.isin(common))
    atac._inplace_subset_obs(atac.obs_names.isin(common))
    if not rna.obs_names.equals(atac.obs_names):
        atac = atac[rna.obs_names].copy()
    mdata.mod["rna"], mdata.mod["atac"] = rna, atac
    mdata.update()


def _parse_peaks(names: pd.Index, include_index: bool = False) -> pd.DataFrame:
    frame = names.to_series(index=np.arange(len(names))).str.extract(
        r"^(?P<Chromosome>[^:]+):(?P<Start>\d+)-(?P<End>\d+)$"
    )
    if frame.isna().any(axis=None):
        bad = names[frame.isna().any(axis=1).to_numpy()][:5].tolist()
        raise ValueError(f"invalid peak names: {bad}")
    frame[["Start", "End"]] = frame[["Start", "End"]].astype(np.int64)
    if include_index:
        frame["original_index"] = np.arange(len(frame), dtype=np.int32)
    return frame


def _merge_atac(samples: list) -> list:
    all_intervals = pd.concat(
        [_parse_peaks(x.mod["atac"].var_names) for x in samples], ignore_index=True
    ).drop_duplicates()
    merged = pr.PyRanges(all_intervals).merge(slack=1).df
    merged["peak"] = (
        merged["Chromosome"].astype(str) + ":" + merged["Start"].astype(str)
        + "-" + merged["End"].astype(str)
    )
    merged["merged_index"] = np.arange(len(merged), dtype=np.int32)
    merged_ranges = pr.PyRanges(merged)
    var = merged.set_index("peak")[["Chromosome", "Start", "End"]]
    result = []
    for mdata in samples:
        atac = mdata.mod["atac"]
        original = _parse_peaks(atac.var_names, include_index=True)
        overlap = pr.PyRanges(original).join(merged_ranges).df
        overlap["overlap_size"] = (
            np.minimum(overlap["End"], overlap["End_b"])
            - np.maximum(overlap["Start"], overlap["Start_b"])
        )
        best = (
            overlap.sort_values(["original_index", "overlap_size"], ascending=[True, False])
            .drop_duplicates("original_index")
        )
        if len(best) != atac.n_vars:
            raise ValueError("one or more original peaks did not map to merged peaks")
        mapping = best.set_index("original_index")["merged_index"].reindex(
            np.arange(atac.n_vars)
        ).to_numpy(dtype=np.int32)
        x = sparse.csr_matrix(atac.X)
        remapped = sparse.csr_matrix(
            (
                x.data.astype(np.float32, copy=False),
                mapping[x.indices],
                x.indptr,
            ),
            shape=(x.shape[0], len(merged)),
        )
        remapped.sum_duplicates()
        new_atac = ad.AnnData(remapped, obs=atac.obs.copy(), var=var.copy())
        result.append(md.MuData({"rna": mdata.mod["rna"], "atac": new_atac}))
    return result


def _set_sample(mdata, sample: str) -> None:
    rna, atac = mdata.mod["rna"], mdata.mod["atac"]
    if not rna.obs_names.equals(atac.obs_names):
        raise ValueError("RNA and ATAC cell order differs")
    prefix = f"{sample}#"
    names = pd.Index(
        [x if str(x).startswith(prefix) else f"{prefix}{x}" for x in rna.obs_names],
        dtype=str,
    )
    if not names.is_unique:
        raise ValueError(f"{sample}: non-unique prefixed cell names")
    for adata in (rna, atac):
        adata.obs_names = names.copy()
        adata.obs["sample"] = pd.Categorical([sample] * adata.n_obs)
    mdata.update()
    mdata.obs["sample"] = pd.Categorical([sample] * mdata.n_obs)


def load_dataset(
    root: Path,
    sample_names: list[str],
    sample_files: list[str] | None = None,
    min_genes: int = 200,
    min_cells: int = 3,
    min_atac_counts: int = 100,
    feature_fraction: float | None = None,
):
    if not root.exists():
        raise FileNotFoundError(root)
    if sample_files is not None and len(sample_files) != len(sample_names):
        raise ValueError("sample_files and sample_names must have equal length")
    if root.is_file() or _is_10x_dir(root) or list(root.glob("*.h5")) or list(root.glob("*.h5mu")):
        if len(sample_names) != 1:
            raise ValueError("single input container requires exactly one metadata sample")
        samples = [(sample_names[0], root)]
    else:
        children = {x.name: x for x in root.iterdir() if x.is_dir()}
        samples = []
        aliases = sample_files or sample_names
        for name, sample_file in zip(sample_names, aliases, strict=True):
            if name in children:
                samples.append((name, children[name]))
                continue
            matches = [path for child, path in children.items() if name in child]
            if len(matches) == 1:
                samples.append((name, matches[0]))
                continue
            if sample_file != name:
                if sample_file in children:
                    LOGGER.info(
                        "sample_path_fallback sample=%s sample_file=%s path=%s",
                        name, sample_file, children[sample_file],
                    )
                    samples.append((name, children[sample_file]))
                    continue
                file_matches = [
                    path for child, path in children.items()
                    if sample_file in child
                ]
                if len(file_matches) == 1:
                    LOGGER.info(
                        "sample_path_fallback sample=%s sample_file=%s path=%s",
                        name, sample_file, file_matches[0],
                    )
                    samples.append((name, file_matches[0]))
                    continue
            raise ValueError(
                f"sample {name!r} (sample_file={sample_file!r}): expected "
                f"one directory match; sample_name_matches="
                f"{[x.name for x in matches]}"
            )
    loaded = []
    for sample, path in samples:
        LOGGER.info("sample_read_start sample=%s path=%s", sample, path)
        current = read_single(path)
        _qc(current, min_genes, min_cells, min_atac_counts)
        _set_sample(current, sample)
        LOGGER.info(
            "sample_read_complete sample=%s cells=%d genes=%d peaks=%d",
            sample, current.mod["rna"].n_obs, current.mod["rna"].n_vars,
            current.mod["atac"].n_vars,
        )
        loaded.append(current)
    if len(loaded) == 1:
        return loaded[0]
    remapped = _merge_atac(loaded)
    del loaded
    gc.collect()
    if feature_fraction is not None:
        total_cells = sum(x.mod["rna"].n_obs for x in remapped)
        threshold = total_cells * feature_fraction
        common_genes = remapped[0].mod["rna"].var_names
        for current in remapped[1:]:
            common_genes = common_genes.intersection(
                current.mod["rna"].var_names, sort=False
            )
        gene_detected = np.zeros(len(common_genes), dtype=np.int64)
        peak_detected = np.zeros(remapped[0].mod["atac"].n_vars, dtype=np.int64)
        for current in remapped:
            rna = current.mod["rna"]
            detected = np.asarray(
                sparse.csr_matrix(rna.X).getnnz(axis=0)
            ).ravel()
            positions = rna.var_names.get_indexer(common_genes)
            gene_detected += detected[positions]
            peak_detected += np.asarray(
                sparse.csr_matrix(current.mod["atac"].X).getnnz(axis=0)
            ).ravel()
        keep_genes = common_genes[gene_detected > threshold]
        keep_peaks = peak_detected > threshold
        LOGGER.info(
            "premerge_feature_filter fraction=%g cells=%d genes=%d/%d "
            "peaks=%d/%d",
            feature_fraction, total_cells, len(keep_genes), len(common_genes),
            int(keep_peaks.sum()), len(keep_peaks),
        )
        for current in remapped:
            current.mod["rna"]._inplace_subset_var(
                current.mod["rna"].var_names.isin(keep_genes)
            )
            current.mod["atac"]._inplace_subset_var(keep_peaks)
        del gene_detected, peak_detected
        gc.collect()
    LOGGER.info("sample_merge_start samples=%d", len(remapped))
    merged = md.concat(remapped, join="inner", merge="unique", index_unique=None)
    del remapped
    gc.collect()
    for adata in [merged, *merged.mod.values()]:
        encoded = adata.obs_names.to_series().str.split("#", n=1).str[0]
        adata.obs["sample"] = pd.Categorical(encoded.to_numpy())
    LOGGER.info(
        "sample_merge_complete cells=%d genes=%d peaks=%d",
        merged.mod["rna"].n_obs, merged.mod["rna"].n_vars,
        merged.mod["atac"].n_vars,
    )
    return merged


class PeakAnnotator:
    def __init__(self, gtf_path: Path):
        genes, transcripts = [], []
        opener = gzip.open if str(gtf_path).endswith(".gz") else open
        with opener(gtf_path, "rt") as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                fields = line.rstrip().split("\t")
                if len(fields) != 9:
                    continue
                chrom, _, feature, start, end, _, strand, _, attrs = fields
                parsed = {}
                for item in attrs.split(";"):
                    item = item.strip()
                    if item and " " in item:
                        key, value = item.split(" ", 1)
                        parsed[key] = value.strip('"')
                if parsed.get("gene_type", parsed.get("gene_biotype")) != "protein_coding":
                    continue
                gene = parsed.get("gene_name", "")
                start, end = int(start), int(end)
                if feature == "gene":
                    genes.append((chrom, start if strand == "+" else end, strand, gene))
                elif feature == "transcript":
                    transcripts.append((chrom, start, end, gene))
        self.genes = pd.DataFrame(genes, columns=["Chromosome", "TSS", "strand", "gene"])
        self.transcripts = pd.DataFrame(
            transcripts, columns=["Chromosome", "Start", "End", "gene"]
        )
        if self.genes.empty:
            raise ValueError(f"no protein-coding genes found in {gtf_path}")
        if not self.genes["Chromosome"].iloc[0].startswith("chr"):
            self.genes["Chromosome"] = "chr" + self.genes["Chromosome"].astype(str)
            self.transcripts["Chromosome"] = "chr" + self.transcripts["Chromosome"].astype(str)

    def annotate(self, peaks: pd.DataFrame) -> pd.DataFrame:
        peaks = peaks.rename(
            columns={"chrom": "Chromosome", "start": "Start", "end": "End"}
        ).copy()
        peak_ranges = pr.PyRanges(peaks)
        promoters = self.genes.copy()
        promoters["Start"] = np.maximum(
            np.where(
                promoters["strand"].eq("+"),
                promoters["TSS"] - 1000,
                promoters["TSS"] - 100,
            ),
            0,
        )
        promoters["End"] = np.where(
            promoters["strand"].eq("+"), promoters["TSS"] + 100, promoters["TSS"] + 1000
        )
        promoter = peak_ranges.join(pr.PyRanges(promoters)).df
        columns = ["Chromosome", "Start", "End", "gene"]
        rows = (
            promoter[columns].copy()
            if not promoter.empty
            else pd.DataFrame(columns=columns)
        )
        rows["distance"], rows["peak_type"] = 0, "promoter"
        seen = set(zip(rows["Chromosome"], rows["Start"], rows["End"], rows["gene"]))
        tss = self.genes.assign(Start=self.genes["TSS"], End=self.genes["TSS"] + 1)
        nearest = peak_ranges.nearest(pr.PyRanges(tss)).df
        distal = []
        for row in nearest.itertuples(index=False):
            key = (row.Chromosome, row.Start, row.End, row.gene)
            point = row.Start_b
            distance = row.Start - point if row.Start > point else row.End - point if row.End < point else 0
            if key not in seen and abs(distance) <= 200_000:
                distal.append((*key, distance, "distal"))
                seen.add(key)
        body = peak_ranges.join(pr.PyRanges(self.transcripts)).df
        for row in body.itertuples(index=False):
            key = (row.Chromosome, row.Start, row.End, row.gene)
            if key not in seen:
                distal.append((*key, 0, "distal"))
                seen.add(key)
        extra = pd.DataFrame(
            distal, columns=["Chromosome", "Start", "End", "gene", "distance", "peak_type"]
        )
        result = pd.concat([rows, extra], ignore_index=True)
        result.index = (
            result["Chromosome"].astype(str) + ":" + result["Start"].astype(str)
            + "-" + result["End"].astype(str)
        )
        return result.sort_values(["Chromosome", "Start", "End"])


def normalize_peak_names(atac):
    """Normalize chromosome prefixes and sum columns that become identical."""
    def normalize(name: str) -> str:
        match = __import__("re").fullmatch(
            r"(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)", name.strip()
        )
        if match is None:
            raise ValueError(f"invalid peak name (expected chrom:start-end): {name}")
        chrom = match.group("chrom")
        rest = f"{match.group('start')}-{match.group('end')}"
        bare = chrom[3:] if chrom.lower().startswith("chr") else chrom
        if bare.upper() in {"M", "MT"}:
            chrom = "chrM"
        elif bare.isdigit() or bare.upper() in {"X", "Y"}:
            chrom = f"chr{bare.upper()}"
        return f"{chrom}:{rest}"
    normalized = pd.Index([normalize(str(x)) for x in atac.var_names])
    changed = int((normalized != atac.var_names.astype(str)).sum())
    if changed:
        LOGGER.info(
            "peak_names_normalized changed=%d total=%d", changed, atac.n_vars
        )
    if normalized.is_unique:
        atac.var_names = normalized
        return atac
    codes, unique_names = pd.factorize(normalized, sort=False)
    matrix = sparse.csr_matrix(atac.X, copy=False)
    collapsed = sparse.csr_matrix(
        (matrix.data, codes[matrix.indices], matrix.indptr),
        shape=(matrix.shape[0], len(unique_names)),
    )
    collapsed.sum_duplicates()
    _, first = np.unique(codes, return_index=True)
    var = atac.var.iloc[first].copy()
    var.index = pd.Index(unique_names)
    LOGGER.info(
        "peak_name_collisions_collapsed columns_before=%d columns_after=%d",
        atac.n_vars, len(unique_names),
    )
    return ad.AnnData(collapsed, obs=atac.obs.copy(), var=var)
