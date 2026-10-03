# Snapshot provenance

The Python modules in `scmore/` are copies of their namesakes from the
working scMORE pipeline. The following portability-only changes were made in
this snapshot:

- server-specific absolute paths were replaced with documented placeholders or
  repository-relative output paths;
- reviewer-facing `README.md`, `requirements.txt`, `environment.yml`, and
  `.gitignore` files were added;
- raw matrices, reference genomes, motif databases, logs, checkpoints, and
  generated backend files were not copied.
- the reviewer metadata is the 1,283-row frontend metadata revision, with a
  matching regenerated `sample-file-map.tsv`.

No working-pipeline source file was modified while creating this snapshot.
