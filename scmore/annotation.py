from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd


REQUIRED_FIELDS = {
    "old_cluster", "cell_type", "cell_ontology_id", "confidence",
    "tissue_consistent", "reason",
}

ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "annotations": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "old_cluster": {"type": "string"},
                    "cell_type": {"type": "string"},
                    "cell_ontology_id": {
                        "type": "string",
                        "pattern": r"^CL:\d{7}$",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "tissue_consistent": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": sorted(REQUIRED_FIELDS),
                "additionalProperties": False,
            },
        }
    },
    "required": ["annotations"],
    "additionalProperties": False,
}


def build_annotation_payload(
    dataset: str, markers: pd.DataFrame, metadata: pd.DataFrame, top_n: int
) -> dict:
    marker_groups = {}
    for cluster, part in markers.groupby(markers["group"].astype(str), sort=False):
        ranked = part.sort_values(["scores", "logfoldchanges"], ascending=False).head(top_n)
        marker_groups[str(cluster)] = ranked[
            ["names", "scores", "logfoldchanges", "pvals_adj"]
        ].where(pd.notna(ranked), None).to_dict("records")
    meta_columns = [
        "sample_name", "organism", "tissueOntologyName", "tissueOntologyID",
        "cellOntologyName", "cellOntologyID", "diseaseOntologyName",
        "treatment", "genotype", "group",
    ]
    return {
        "dataset": dataset,
        "sample_metadata": metadata[[c for c in meta_columns if c in metadata]].fillna("notAvailable").to_dict("records"),
        "clusters": marker_groups,
    }


def _prompt(payload: dict) -> str:
    return """You are annotating single-cell multiome RNA clusters.
Use every cluster's ranked marker genes and the sample metadata. Return one
annotation per cluster. `cell_type` MUST be the canonical English Cell Ontology
label and `cell_ontology_id` MUST be a matching CL:NNNNNNN identifier. Check
whether the proposed type is biologically plausible in the supplied tissue.
Use a broader valid ontology parent when subtype evidence is insufficient.
Never invent an ontology identifier. Keep `old_cluster` exactly as supplied.
Return JSON only with this shape:
{"annotations":[{"old_cluster":"0","cell_type":"...","cell_ontology_id":"CL:...",
"confidence":"high|medium|low","tissue_consistent":true,"reason":"brief marker/tissue evidence"}]}

INPUT:
""" + json.dumps(payload, ensure_ascii=False)


def request_annotations(
    payload: dict,
    model: str,
    response_file: Path | None = None,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    mode: str = "openai",
    codex_executable: str = "codex",
    codex_work_dir: Path | None = None,
) -> dict:
    if response_file:
        return json.loads(response_file.read_text())
    if mode == "codex":
        work_dir = codex_work_dir or Path.cwd()
        work_dir.mkdir(parents=True, exist_ok=True)
        schema_path = work_dir / "annotation_schema.json"
        output_path = work_dir / "codex_annotation_response.json"
        schema_path.write_text(
            json.dumps(ANNOTATION_SCHEMA, ensure_ascii=False, indent=2)
        )
        command = [
            codex_executable, "--ask-for-approval", "never", "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox", "read-only",
            "--output-schema", str(schema_path.resolve()),
            "--output-last-message", str(output_path.resolve()),
            "--color", "never",
            "-",
        ]
        try:
            subprocess.run(
                command,
                input=_prompt(payload),
                text=True,
                cwd=work_dir,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Codex executable not found: {codex_executable}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Codex annotation failed with exit code {exc.returncode}"
            ) from exc
        return json.loads(output_path.read_text())
    if mode != "openai":
        raise ValueError(f"unsupported annotation mode: {mode}")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Install the openai package or provide annotation_response in config"
        ) from exc
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is unset; cannot run GPT annotation")
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.responses.create(
        model=model,
        input=_prompt(payload),
        text={"format": {"type": "json_object"}},
    )
    return json.loads(response.output_text)


def validate_annotations(response: dict, clusters: set[str]) -> pd.DataFrame:
    rows = response.get("annotations")
    if not isinstance(rows, list) or not rows:
        raise ValueError("annotation JSON must contain a non-empty annotations list")
    frame = pd.DataFrame(rows)
    missing = REQUIRED_FIELDS - set(frame.columns)
    if missing:
        raise ValueError(f"annotation fields missing: {sorted(missing)}")
    frame["old_cluster"] = frame["old_cluster"].astype(str)
    if frame["old_cluster"].duplicated().any() or set(frame["old_cluster"]) != clusters:
        raise ValueError("annotation cluster set does not exactly match observed clusters")
    valid_id = frame["cell_ontology_id"].astype(str).str.fullmatch(r"CL:\d{7}")
    if not valid_id.all():
        raise ValueError("all cell_ontology_id values must match CL:NNNNNNN")
    if not frame["confidence"].isin(["high", "medium", "low"]).all():
        raise ValueError("confidence must be high, medium, or low")
    if not frame["tissue_consistent"].map(lambda x: isinstance(x, bool)).all():
        raise ValueError("tissue_consistent must be boolean")
    return frame.sort_values(
        "old_cluster", key=lambda s: s.map(lambda x: int(x) if re.fullmatch(r"\d+", x) else x)
    )


def annotate_mdata(
    mdata, mapping: pd.DataFrame, json_path: Path, mapping_path: Path
) -> None:
    lookup = mapping.set_index("old_cluster")
    rename = {
        "cell_type": "gpt_cell_type",
        "cell_ontology_id": "gpt_cell_type_id",
        "confidence": "gpt_annotation_confidence",
        "reason": "gpt_annotation_evidence",
    }
    for adata in [mdata, *mdata.mod.values()]:
        clusters = adata.obs["leiden"].astype(str)
        for source, target in rename.items():
            adata.obs[target] = pd.Categorical(clusters.map(lookup[source]))
    mapping.to_csv(mapping_path, index=False)
    json_path.write_text(
        json.dumps({"annotations": mapping.to_dict("records")}, ensure_ascii=False, indent=2)
    )
    mdata.uns["gpt_cell_annotation"] = {
        "mapping_file": mapping_path.name,
        "json_file": json_path.name,
        "basis": "cluster top markers plus sample metadata and tissue consistency check",
    }
