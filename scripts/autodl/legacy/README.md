# Legacy AutoDL scripts

This directory preserves the numbered and one-off launchers used by earlier
experiments. They retain their historical source for comparison and provenance
review, but their old top-level paths are intentionally retired. Local `.orig`
patch backups are ignored and are not part of the tracked archive.

Use the consolidated entrypoints in the parent directory for new work:

```text
../prepare_data.sh
../train_sft.sh
../train_rl.sh
../benchmark.sh
```

The original 500-line command reference is archived as
`README.numbered-scripts.md`. Existing output directories, manifests, logs, and
run summaries are unchanged.
