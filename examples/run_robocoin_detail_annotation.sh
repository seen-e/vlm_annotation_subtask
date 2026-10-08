#!/usr/bin/env bash
cd /mnt/workspace/vlm_auto_annotatiion_refactor
if [ -f /opt/conda/bin/activate ]; then
  source /opt/conda/bin/activate base
fi
mkdir -p /mnt/workspace/vlm_auto_annotatiion_refactor_robocin/examples/outputs

nohup python /mnt/workspace/vlm_auto_annotatiion_refactor/examples/main.py \
  --config /mnt/workspace/vlm_auto_annotatiion_refactor_robocin/config/config_annotation.yaml \
  --workers 32 \
  --skip-existing \
  --scan-workers 128 \
  --start-index 0 \
  --limit 160000 \
  --tasks /mnt/workspace/vlm_auto_annotatiion_refactor_robocin/examples/test_data/robocoin_missing_tasks_with_annotations_preferred_head_view_subtask_splits.nonempty.json \
  --output-dir /mnt/data/chachaxu/dataset/robocoin_detail_annotation_filter \
  > /mnt/workspace/vlm_auto_annotatiion_refactor_robocin/examples/outputs/robocoin_detail_annotation.log 2>&1 &
