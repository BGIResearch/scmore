from __future__ import annotations

from pathlib import Path

import pandas as pd


COLUMN_MAPPING = {
    "PROJID": "project_id",
    "SAMID": "sample_id",
    "sample_name": "sample_name",
    "organism": "biosource_organism",
    "tissueOntologyName": "biosource_tissue",
    "tissueOntologyID": "biosource_tissue_ontology_id",
    "isCellLine": "biosource_is_cell_line",
    "isOrganoid": "biosource_is_organoid",
    "cellLineOntologyName": "biosource_cell_line",
    "cellLineOntologyID": "biosource_cell_line_ontology_id",
    "cellOntologyName": "biosource_cell_type",
    "cellOntologyID": "biosource_cell_type_ontology_id",
    "diseaseOntologyName": "biosource_disease",
    "diseaseOntologyID": "biosource_disease_ontology_id",
    "treatment": "experiment_treatment",
    "genotype": "experiment_genotype",
    "group": "experiment_group",
    "donor": "donor_id",
    "sex": "donor_sex",
    "age": "donor_age",
    "age_group": "donor_age_group",
}

ORGANISM_DATASET_SUFFIXES = {
    "_Homo_sapiens": "Homo sapiens",
    "_Mus_musculus": "Mus musculus",
}


def dataset_source_and_organism(dataset: str) -> tuple[str, str | None]:
    """Resolve a logical split-dataset name to its source GSE and organism."""
    for suffix, organism in ORGANISM_DATASET_SUFFIXES.items():
        if dataset.endswith(suffix):
            source = dataset[: -len(suffix)]
            if not source:
                raise ValueError(f"invalid organism-split dataset name: {dataset}")
            return source, organism
    return dataset, None


def read_dataset_metadata(
    path: Path, dataset: str, sample_file_map: Path | None = None
) -> pd.DataFrame:
    frame = pd.read_excel(path, dtype=object) if path.suffix in {".xls", ".xlsx"} else pd.read_csv(path, dtype=object)
    required = {"gse", *COLUMN_MAPPING}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"metadata missing columns: {missing}")
    source, selected_organism = dataset_source_and_organism(dataset)
    selected = frame["gse"].astype(str).eq(source)
    if selected_organism is not None:
        selected &= frame["organism"].astype(str).eq(selected_organism)
    frame = frame.loc[selected, ["gse", *COLUMN_MAPPING]].copy()
    if frame.empty:
        raise ValueError(f"metadata has no rows for {dataset}")
    if sample_file_map is not None:
        mapping = pd.read_csv(sample_file_map, sep="\t", dtype=object)
        required_mapping = {"gse", "SAMID", "sample_file"}
        missing_mapping = required_mapping - set(mapping)
        if missing_mapping:
            raise ValueError(
                f"sample file map missing columns: {sorted(missing_mapping)}"
            )
        mapping = mapping[["gse", "SAMID", "sample_file"]].copy()
        if mapping.duplicated(["gse", "SAMID"]).any():
            raise ValueError("sample file map has duplicate gse/SAMID rows")
        frame = frame.merge(
            mapping, on=["gse", "SAMID"], how="left", validate="one_to_one"
        )
        missing_files = frame["sample_file"].isna()
        if missing_files.any():
            values = frame.loc[missing_files, "SAMID"].astype(str).tolist()
            raise ValueError(
                f"{dataset}: sample_file missing for SAMID values: {values}"
            )
    frame["sample_name"] = frame["sample_name"].astype(str).str.strip()
    if "sample_file" in frame:
        frame["sample_file"] = frame["sample_file"].astype(str).str.strip()
    if frame["sample_name"].duplicated().any():
        values = frame.loc[frame["sample_name"].duplicated(False), "sample_name"].tolist()
        raise ValueError(f"{dataset}: duplicate sample_name values: {values}")
    return frame


def add_metadata(mdata, metadata: pd.DataFrame) -> None:
    lookup = metadata.set_index("sample_name")
    expected = set(lookup.index)
    for name, adata in [("mudata", mdata), *mdata.mod.items()]:
        if "sample" not in adata.obs:
            if len(lookup) != 1:
                raise ValueError(f"{name}: no sample column for multi-sample metadata")
            samples = pd.Series(lookup.index[0], index=adata.obs_names)
        else:
            samples = adata.obs["sample"].astype(str)
        observed = set(samples)
        if observed != expected:
            raise ValueError(
                f"{name}: sample mismatch; data_only={sorted(observed-expected)}, "
                f"meta_only={sorted(expected-observed)}"
            )
        for source, target in COLUMN_MAPPING.items():
            values = samples if source == "sample_name" else samples.map(lookup[source])
            # anndata versions used by this pipeline cannot serialize
            # pandas StringArray-backed categorical categories to HDF5.
            normalized = (
                values.fillna("notAvailable").astype(str).to_numpy(dtype=object)
            )
            adata.obs[target] = pd.Categorical(normalized)


def dataset_organism(metadata: pd.DataFrame) -> str:
    organisms = metadata["organism"].dropna().astype(str).unique()
    if len(organisms) != 1:
        raise ValueError(f"expected one organism, got {organisms.tolist()}")
    return organisms[0]
