# scMORE multiprocessor

每个 dataset 由一个独立进程完成：读取和 >1% feature 过滤、peak 注释、meta
写入、RNA clustering/marker、GPT Cell Ontology 注释、RNA/ATAC UMAP、
pseudobulk peak–gene、TF–peak–gene、active score、H5MU 和 backend parquet。

运行：

```bash
export OPENAI_API_KEY=...
python -m scmore.multiprocessor --config scmore/config.example.json
```

配置 `"annotation_mode": "codex"` 时通过已登录的 Codex CLI 做注释，不需要
`OPENAI_API_KEY`；`"annotation_mode": "openai"` 则使用 OpenAI API。

也可以预先人工审核 GPT JSON，然后在配置中设置
`"annotation_response": "/path/reviewed.json"`，此时不会访问 API。JSON 必须包含
每个 cluster 的标准 Cell Ontology name、CL ID、置信度、组织合理性和理由。

FIMO 已集成到流程。默认使用配置中按 organism 指定的 genome FASTA 和
CisDB/CIS-BP 3.10 目录。人类使用 `ref/motif/hs`，小鼠使用
`ref/motif/mmu`。流程从 `TF_Information_all_motifs_plus.txt` 和
`pwms_all_motifs/*.txt` 自动选择代表 PWM、生成 MEME，再对所有保留 ATAC
peaks 扫描，并生成
`work_root/dataset/fimo/tf_peak_hits.parquet`。`motif_hits[dataset]` 仅用于
显式复用已有结果；留空 `{}` 即自动运行 FIMO。
`fimo_jobs` 是并发上限；实际并发会根据细胞数和运行时可用内存动态下调，
并在 dataset 日志中记录 `fimo_concurrency`。

输出位置为 `output_root/dataset/`，H5MU 固定为
`dataset/dataset.h5mu`。cell-level parquet 使用固定 seed，按
`sample_id × gpt_cell_type` 分层降采样；peak–gene、triplet 保留完整结果，
`regulatory/triplet_active_scores.parquet` 明确不降采样。
导出完成后，流程使用 DuckDB 将降采样后的 ATAC 表与完整 peak–gene links
连接并聚合，原子写入 `expression/gene_activity.parquet`，字段为
`gene、cell_id、activity`。
在 cell-level 降采样之前，流程还会从完整稀疏矩阵分批计算 zero-aware
context means，写入 `expression/gene_mean_context.parquet` 和
`expression/peak_mean_context.parquet`。均包含 `context_name`、
`context_value`、feature、`mean_value`、`n_cells` 和 `nnz`；未存储的稀疏
值按 0 计入均值分母。

数据读取、QC、跨样本 peak merge、GTF peak annotation 均实现在当前目录的
`data_processing.py`，不依赖旧版 `scmore.MultiomePipeline`。本版本不计算 WNN。

设置 `run_rapids=true` 后使用当前目录的 `run_rapids_h5mu_compat.py`
计算 RNA/ATAC PCA、neighbors 和 UMAP。该脚本不计算 WNN，并会保护
`rank_genes_groups_wilcoxon` recarray、校验输出后再替换原 H5MU。

断点续跑默认开启。H5MU 使用 `.building` 临时文件后原子替换；失败数据集不会影响
其他 worker，汇总写入 `work_root/multiprocessor_report.json`。

日志使用 Python `logging` 和滚动文件：

- `work_root/multiprocessor.log`：总控、worker 结果和失败汇总；
- `work_root/dataset/pipeline.log`：单 dataset 全阶段日志；
- 单文件最大 50 MB，保留 5 个备份；
- 格式包含时间、级别、dataset、PID、模块、阶段耗时和异常堆栈。

配置项 `log_level` 默认为 `INFO`，可改为 `DEBUG`、`WARNING` 或 `ERROR`。
