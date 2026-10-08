# examples/run_pipeline.py

## 文件职责

命令行运行完整 pipeline 的示例脚本。适合执行 `scene -> analysis -> refinement` 或从中间 stage 继续执行。

## 参数

| 参数 | 是否必需 | 说明 |
|---|---:|---|
| `--config` | 否 | 配置文件路径 |
| `--video` | 是 | 视频路径 |
| `--instruction` | 是 | 任务指令 |
| `--video-id` | 否 | 视频/episode id |
| `--dry-run` | 否 | 不调用模型 |
| `--start-from` | 否 | 从指定 stage 开始 |
| `--stop-after` | 否 | 执行到指定 stage 后停止 |
| `--skip-existing` | 否 | 跳过 context 中已有 stage |
| `--save-results` | 否 | 保存中间结果 |
| `--output-dir` | 否 | 输出根目录 |
| `--run-name` | 否 | 当前运行名称 |
| `--load-run-dir` | 否 | 加载已有运行目录 |
| `--load-stages` | 否 | 要从已有运行目录加载的 stage 列表 |

## 示例：完整 dry-run

```bash
python vlm_auto_annotation_refactor/examples/run_pipeline.py   --video /path/to/video.mp4   --instruction "pick up the cup"   --dry-run   --save-results
```

## 示例：加载 scene，只跑 analysis 后续

```bash
python vlm_auto_annotation_refactor/examples/run_pipeline.py   --video /path/to/video.mp4   --instruction "pick up the cup"   --load-run-dir outputs/debug_pipeline   --load-stages scene   --start-from analysis   --skip-existing   --dry-run
```
