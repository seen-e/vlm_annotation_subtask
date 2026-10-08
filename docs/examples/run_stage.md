# examples/run_stage.py

## 文件职责

命令行运行单个 stage 的示例脚本。适合调试某个 stage 的视频处理和 prompt 渲染。

## 参数

| 参数 | 是否必需 | 说明 |
|---|---:|---|
| `--config` | 否 | 配置文件路径，默认 `vlm_auto_annotation_refactor/config.yaml` |
| `--stage` | 否 | stage 名称，默认 `scene` |
| `--video` | 是 | 视频路径 |
| `--instruction` | 是 | 任务指令 |
| `--video-id` | 否 | 视频/episode id |
| `--dry-run` | 否 | 不调用模型，仅测试视频处理和 prompt 渲染 |
| `--save-results` | 否 | 保存结果到 outputs |
| `--output-dir` | 否 | 输出根目录 |
| `--run-name` | 否 | 当前运行名称 |

## 示例

```bash
python vlm_auto_annotation_refactor/examples/run_stage.py   --stage scene   --video /path/to/video.mp4   --instruction "pick up the cup"   --dry-run   --save-results
```
