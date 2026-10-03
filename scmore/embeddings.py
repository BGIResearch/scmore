from __future__ import annotations

import numpy as np
import scanpy as sc


def strip_wnn(mdata) -> bool:
    """Remove WNN artifacts from files produced by an older pipeline version."""
    changed = False
    for key in ("wnn_umap", "wnn_weights"):
        if key in mdata.obsm:
            del mdata.obsm[key]
            changed = True
    for key in ("wnn_connectivities", "wnn_distances"):
        if key in mdata.obsp:
            del mdata.obsp[key]
            changed = True
    if "wnn" in mdata.uns:
        del mdata.uns["wnn"]
        changed = True
    return changed


def add_umaps(mdata, n_pcs: int = 50, n_neighbors: int = 20, seed: int = 0) -> None:
    rna, atac = mdata.mod["rna"], mdata.mod["atac"]
    if not rna.obs_names.equals(atac.obs_names):
        raise ValueError("RNA and ATAC cell order differs")
    if "highly_variable" not in rna.var:
        sc.pp.highly_variable_genes(rna, n_top_genes=min(3000, rna.n_vars))
    rna_analysis = rna[:, rna.var["highly_variable"].astype(bool)].copy()
    sc.pp.scale(rna_analysis, zero_center=False)
    npc = min(n_pcs, rna_analysis.n_obs - 1, rna_analysis.n_vars - 1)
    sc.pp.pca(
        rna_analysis, n_comps=npc, random_state=seed, zero_center=False
    )
    sc.pp.neighbors(rna_analysis, n_neighbors=n_neighbors, random_state=seed)
    sc.tl.umap(rna_analysis, random_state=seed)
    rna.obsm["X_pca"] = np.asarray(rna_analysis.obsm["X_pca"], dtype=np.float32)
    rna.obsm["X_umap"] = np.asarray(rna_analysis.obsm["X_umap"], dtype=np.float32)
    rna.obsp["distances"] = rna_analysis.obsp["distances"].copy()
    rna.obsp["connectivities"] = rna_analysis.obsp["connectivities"].copy()

    hvg = set(rna.var_names[rna.var["highly_variable"].astype(bool)])
    ann = mdata.uns["peak_annotation"]
    linked = set(ann.index[ann["gene"].isin(hvg)])
    mask = np.asarray(atac.var_names.isin(linked))
    if not mask.any():
        raise ValueError("no ATAC peaks link to RNA HVGs")
    atac.var["peak_in_rna_hvg"] = mask
    atac_analysis = atac[:, mask].copy()
    apc = min(n_pcs, atac_analysis.n_obs - 1, atac_analysis.n_vars - 1)
    sc.pp.pca(
        atac_analysis, n_comps=apc, random_state=seed, zero_center=False
    )
    sc.pp.neighbors(atac_analysis, n_neighbors=n_neighbors, random_state=seed)
    sc.tl.umap(atac_analysis, random_state=seed)
    atac.obsm["X_pca"] = np.asarray(atac_analysis.obsm["X_pca"], dtype=np.float32)
    atac.obsm["X_umap"] = np.asarray(atac_analysis.obsm["X_umap"], dtype=np.float32)
    atac.obsp["distances"] = atac_analysis.obsp["distances"].copy()
    atac.obsp["connectivities"] = atac_analysis.obsp["connectivities"].copy()
