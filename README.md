# scMORE multiome processing pipeline

This directory is a reviewer-oriented snapshot of the scMORE processing code.
It contains the complete shared pipeline used to process paired scRNA-seq and
scATAC-seq datasets, plus the metadata tables and post-processing utilities used
to calculate and filter cell-level TF–peak–gene activity.

The snapshot is independent of the working pipeline: editing files here does
not modify the original working files.

## Included workflow

1. Read paired RNA/ATAC matrices and perform QC and feature filtering.
2. Normalize peak names, merge samples, and annotate peaks using a GTF file.
3. Add curated dataset/sample metadata.
4. Cluster cells, calculate markers, and request Cell Ontology annotations.
5. Calculate RNA and ATAC PCA/neighbors/UMAP using RAPIDS when enabled, with a
   sparse CPU fallback.
6. Calculate pseudobulk peak–gene links.
7. Scan accessible peaks for TF motifs with FIMO.
8. Construct TF–peak–gene triplets and pseudobulk activity scores.
9. Export H5MU and backend Parquet files, including context means and gene
   activity.
10. Calculate the fraction of cells jointly detecting TF RNA, target RNA, and
    peak accessibility, then optionally filter the results at dataset level.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── data
│   ├── scmore-meta-0727-with-data.frontend.csv
│   └── sample-file-map.tsv
└── scmore
    ├── config.example.json
    ├── multiprocessor.py
    └── ... supporting modules
```

## External requirements

The repository intentionally does not include controlled/large inputs:

- paired RNA/ATAC input matrices;
- genome FASTA and gene GTF files;
- CIS-BP motif directories;
- FIMO from MEME Suite and `bedtools`;
- a RAPIDS environment (optional);
- Codex CLI login or an OpenAI API key for automated annotation.

Update the placeholder paths in `scmore/config.example.json` before running.

The metadata snapshot contains 1,283 sample rows. `sample-file-map.tsv` also
contains 1,283 one-to-one `gse + SAMID` mappings. The demultiplexed HTO rows for
GSE247442 and GSE251978 map back to their respective parent input containers;
their final sample labels and experimental groups are retained in the metadata.

## Run the complete pipeline

From the repository root:

```bash
python -m scmore.multiprocessor \
  --config scmore/config.example.json
```

To process selected datasets or override concurrency:

```bash
python -m scmore.multiprocessor \
  --config scmore/config.example.json \
  --datasets GSE251978 GSE247442 \
  --workers 2 \
  --fimo-jobs 8
```

The pipeline uses per-dataset locks, atomic H5MU writes, stage checkpoints, and
`resume=true` by default.

## Cell-level activity fraction

After the main pipeline has generated `triplet_active_scores.parquet`, add the
joint detection fields without overwriting the original file:

```bash
python -m scmore.add_active_cell_fraction \
  --backend-root /path/to/scmore_backend_v2
```

This creates `triplet_active_scores.with_active_cells.parquet`, including:

- `active_cell_count`;
- `group_total_cells`;
- `active_cell_fraction`.

A cell is active when TF RNA, target-gene RNA, and peak accessibility are all
strictly greater than zero.

## Dataset-level filtering

The published review snapshot supports configurable filtering. For example,
the exploratory `>1%` and `>20 active cells` result is generated with:

```bash
python -m scmore.filter_active_cell_scores \
  --backend-root /path/to/scmore_backend_v2 \
  --summary /path/to/active_cell_filter_gt1pct_n20_summary.tsv \
  --score-threshold 0.01 \
  --count-threshold 20 \
  --output-name triplet_active_scores.dataset_active_gt1pct_n20.parquet
```

Filtering is performed after pooling sample-level counts within each dataset,
cell type, TF, peak, and target gene.

## Reproducibility notes

- Cell-level visualization exports are stratified and capped by
  `export_max_cells`; regulatory tables are not downsampled.
- Random operations use `random_seed` from the configuration.
- The pipeline records a global multiprocessor log and one log per dataset.
- Dataset-specific exploratory rerun scripts are not part of the shared main
  workflow and are therefore excluded from this reviewer snapshot.
