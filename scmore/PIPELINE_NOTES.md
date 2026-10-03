# scMORE multiprocessor

Each dataset is processed in an independent worker process. The workflow reads
the input matrices, retains features detected in more than 1% of cells,
annotates peaks, attaches metadata, performs RNA clustering and marker
analysis, requests GPT-assisted Cell Ontology annotations, calculates RNA and
ATAC UMAP embeddings, constructs pseudobulk peak–gene links and TF–peak–gene
triplets, calculates activity scores, and exports H5MU and backend Parquet
files.

## Running the pipeline

```bash
export OPENAI_API_KEY=...
python -m scmore.multiprocessor --config scmore/config.example.json
```

When `"annotation_mode": "codex"`, annotation is performed through an
authenticated Codex CLI session and `OPENAI_API_KEY` is not required. Set
`"annotation_mode": "openai"` to use the OpenAI API instead.

Annotation responses may also be reviewed manually in advance. Set
`"annotation_response": "/path/to/reviewed.json"` in the configuration to use
the reviewed file without contacting an API. The JSON must provide, for every
cluster, a canonical Cell Ontology label, matching CL identifier, confidence,
tissue-consistency flag, and supporting rationale.

## FIMO motif scanning

FIMO is integrated into the workflow. By default, it uses the organism-specific
genome FASTA and CisDB/CIS-BP 3.10 directory configured by the user. The
workflow reads `TF_Information_all_motifs_plus.txt` and
`pwms_all_motifs/*.txt`, selects representative PWMs, creates a MEME motif
file, scans every retained ATAC peak, and writes:

```text
work_root/dataset/fimo/tf_peak_hits.parquet
```

`motif_hits[dataset]` is only needed to reuse a precomputed result explicitly.
Leave `motif_hits` as `{}` to run FIMO automatically. `fimo_jobs` defines the
maximum concurrency. The effective concurrency is reduced dynamically based on
the number of cells and available memory and is recorded as `fimo_concurrency`
in the dataset log.

## Outputs and downsampling

Results are written below `output_root/dataset/`, and the H5MU path is always:

```text
output_root/dataset/dataset.h5mu
```

Cell-level Parquet tables are stratified by
`sample_id × gpt_cell_type` and downsampled with a fixed random seed. Complete
peak–gene links and TF–peak–gene triplets are retained.
`regulatory/triplet_active_scores.parquet` is explicitly not downsampled.

After export, the workflow uses DuckDB to join downsampled ATAC values with the
complete peak–gene links, aggregates the results, and atomically writes
`expression/gene_activity.parquet` with the columns `gene`, `cell_id`, and
`activity`.

Before cell-level downsampling, zero-aware context means are calculated in
batches from the complete sparse matrices and written to:

```text
expression/gene_mean_context.parquet
expression/peak_mean_context.parquet
```

Both tables contain `context_name`, `context_value`, the corresponding feature,
`mean_value`, `n_cells`, and `nnz`. Values omitted from sparse storage are
treated as zero in the mean denominator.

## Data processing and embeddings

Input reading, quality control, cross-sample peak merging, and GTF-based peak
annotation are implemented in `data_processing.py`. The workflow does not
depend on the legacy `scmore.MultiomePipeline` and does not calculate WNN.

When `run_rapids=true`, `run_rapids_h5mu_compat.py` calculates RNA and ATAC PCA,
nearest-neighbor graphs, and UMAP embeddings. It does not calculate WNN. The
compatibility wrapper preserves the `rank_genes_groups_wilcoxon` record array,
validates the generated embeddings, and only then replaces the original H5MU.
If GPU processing fails, the main workflow falls back to the sparse CPU
implementation.

## Checkpoints and logging

Checkpoint resumption is enabled by default. H5MU files are first written to a
`.building` temporary path and then replaced atomically. A failed dataset does
not interrupt other workers. The batch summary is written to:

```text
work_root/multiprocessor_report.json
```

Logging uses Python `logging` with rotating files:

- `work_root/multiprocessor.log` contains controller messages, worker results,
  and the failure summary.
- `work_root/dataset/pipeline.log` contains all stages for one dataset.
- Each log file is limited to 50 MB, with five backups retained.
- Records include time, severity, dataset, PID, module, stage duration, and
  exception stack traces.

`log_level` defaults to `INFO` and may be changed to `DEBUG`, `WARNING`, or
`ERROR`.
