from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .data_processing import GTF_BY_ORGANISM

DEFAULT_META = (
    Path(__file__).resolve().parents[1]
    / "data/scmore-meta-0727-with-data.frontend.csv"
)
DEFAULT_SAMPLE_FILE_MAP = (
    Path(__file__).resolve().parents[1] / "data/sample-file-map.tsv"
)
DEFAULT_OUTPUT_ROOT = Path("./output/scmore_backend_v2")
REFERENCE_ROOT = Path("/path/to/reference_data")


@dataclass(frozen=True)
class Thresholds:
    feature_fraction: float = 0.01
    min_pseudobulk_cells: int = 10
    peak_gene_fdr: float = 0.05
    tf_pseudobulk_mean: float = 0.5
    peak_gene_r: float = 0.5
    motif_pvalue: float = 1e-5
    tf_gene_r: float = 0.5
    tf_pseudobulk_max: float = 0.5


@dataclass
class PipelineConfig:
    data_root: Path
    datasets: list[str]
    metadata: Path = DEFAULT_META
    sample_file_map: Path | None = DEFAULT_SAMPLE_FILE_MAP
    output_root: Path = DEFAULT_OUTPUT_ROOT
    work_root: Path = Path("./work")
    workers: int = 1
    gtf_by_organism: dict[str, Path] = field(
        default_factory=lambda: GTF_BY_ORGANISM.copy()
    )
    genome_fasta_by_organism: dict[str, Path] = field(
        default_factory=lambda: {
            "Homo sapiens": REFERENCE_ROOT / "refdata-gex-GRCh38-2024-A/fasta/genome.fa",
            "Mus musculus": REFERENCE_ROOT / "refdata-cellranger-arc-mm10-2020-A-2.0.0/fasta/genome.fa",
            "Drosophila melanogaster": REFERENCE_ROOT / "refdata-scmore-Drosophila_melanogaster-BDGP6.54/fasta/genome.fa",
            "Danio rerio": REFERENCE_ROOT / "refdata-scmore-Danio_rerio-GRCz11/fasta/genome.fa",
            "Rattus norvegicus": REFERENCE_ROOT / "refdata-scmore-Rattus_norvegicus-GRCr8/fasta/genome.fa",
            "Gallus gallus": REFERENCE_ROOT / "refdata-scmore-Gallus_gallus-GRCg7b/fasta/genome.fa",
        }
    )
    genome_fasta_by_dataset: dict[str, Path] = field(default_factory=dict)
    gtf_by_dataset: dict[str, Path] = field(default_factory=dict)
    cisdb_by_organism: dict[str, Path] = field(
        default_factory=lambda: {
            "Homo sapiens": Path(__file__).resolve().parents[2] / "ref/motif/hs",
            "Mus musculus": Path(__file__).resolve().parents[2] / "ref/motif/mmu",
            "Drosophila melanogaster": REFERENCE_ROOT / "refdata-scmore-Drosophila_melanogaster-BDGP6.54/cisbp",
            "Danio rerio": REFERENCE_ROOT / "refdata-scmore-Danio_rerio-GRCz11/cisbp",
            "Rattus norvegicus": REFERENCE_ROOT / "refdata-scmore-Rattus_norvegicus-GRCr8/cisbp",
            "Gallus gallus": REFERENCE_ROOT / "refdata-scmore-Gallus_gallus-GRCg7b/cisbp",
        }
    )
    fimo_jobs: int = 4
    fimo_sequences_per_chunk: int = 2_000
    fimo_executable: str = "fimo"
    bedtools_executable: str = "bedtools"
    top_markers: int = 30
    annotation_mode: str = "openai"
    annotation_model: str = "gpt-5"
    annotation_base_url: str | None = None
    annotation_api_key_env: str = "OPENAI_API_KEY"
    annotation_response: Path | None = None
    codex_executable: str = "codex"
    motif_hits: dict[str, Path] = field(default_factory=dict)
    resume: bool = True
    run_rapids: bool = False
    rapids_env: Path = Path("/path/to/rapids_singlecell_env")
    gpu_ids: list[str] = field(default_factory=lambda: ["0"])
    export_max_cells: int = 100_000
    export_min_per_stratum: int = 20
    random_seed: int = 0
    log_level: str = "INFO"
    thresholds: Thresholds = field(default_factory=Thresholds)

    def __post_init__(self) -> None:
        for name in ("data_root", "metadata", "output_root", "work_root", "rapids_env"):
            setattr(self, name, Path(getattr(self, name)).expanduser())
        if self.sample_file_map is not None:
            self.sample_file_map = Path(self.sample_file_map).expanduser()
        self.motif_hits = {k: Path(v).expanduser() for k, v in self.motif_hits.items()}
        self.gtf_by_organism = {
            k: Path(v).expanduser() for k, v in self.gtf_by_organism.items()
        }
        for name in (
            "genome_fasta_by_organism", "genome_fasta_by_dataset",
            "gtf_by_dataset", "cisdb_by_organism",
        ):
            setattr(
                self, name,
                {k: Path(v).expanduser() for k, v in getattr(self, name).items()},
            )
        if self.annotation_response:
            self.annotation_response = Path(self.annotation_response).expanduser()
        if not self.datasets:
            raise ValueError("datasets must not be empty")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.annotation_mode not in {"openai", "codex"}:
            raise ValueError("annotation_mode must be 'openai' or 'codex'")

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        import json

        raw: dict[str, Any] = json.loads(Path(path).read_text())
        raw["thresholds"] = Thresholds(**raw.get("thresholds", {}))
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("data_root", "metadata", "output_root", "work_root", "rapids_env"):
            result[key] = str(result[key])
        if result["sample_file_map"] is not None:
            result["sample_file_map"] = str(result["sample_file_map"])
        result["motif_hits"] = {k: str(v) for k, v in self.motif_hits.items()}
        result["gtf_by_organism"] = {
            k: str(v) for k, v in self.gtf_by_organism.items()
        }
        for key in (
            "genome_fasta_by_organism", "genome_fasta_by_dataset",
            "gtf_by_dataset", "cisdb_by_organism",
        ):
            result[key] = {k: str(v) for k, v in getattr(self, key).items()}
        if result["annotation_response"] is not None:
            result["annotation_response"] = str(result["annotation_response"])
        return result

    def genome_fasta(self, dataset: str, organism: str) -> Path:
        return self.genome_fasta_by_dataset.get(
            dataset, self.genome_fasta_by_organism[organism]
        )

    def gtf(self, dataset: str, organism: str) -> Path:
        return self.gtf_by_dataset.get(dataset, self.gtf_by_organism[organism])
