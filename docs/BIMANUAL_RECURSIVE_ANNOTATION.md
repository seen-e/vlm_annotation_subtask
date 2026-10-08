# 双臂 Episode 递归标注、切分与时间线拼接方案

本文档说明当前三阶段标注框架如何对一个完整 episode 进行递归分段预测，如何为 `arm_1`、`arm_2` 维护彼此独立的动作时间线，以及如何在所有片段预测完成后生成最终 `merged_timeline`。

本文描述的是当前代码中的实际行为，主要对应以下模块：

- `episode_runner.py`：完整 episode 的递归调度、断点状态和最终落盘。
- `segment_postprocessor.py`：片段结果校验、下一片段入口选择、逐臂时间线拼接。
- `stage_runner.py`：片段时间窗、逐臂标注契约和真实采样帧信息注入。
- `postprocessors/analysis_boundary_union.py`：最终 subtask ID 分配和双臂边界并集。
- `prompts/prompts_3_stages/analysis.py`：逐臂动作识别与粗边界。
- `prompts/prompts_3_stages/refinement.py`：逐臂边界精修。

## 1. 目标与核心原则

完整 episode 可能远长于一次适合发送给 VLM 的视频长度，因此框架使用固定长度的视频窗口重复运行完整三阶段 pipeline：

```text
Scene -> Analysis -> Refinement
```
当前默认窗口长度为 15 秒。每一轮都重新运行三阶段，而不是只在第一轮运行 Scene，或只对后续窗口运行 Analysis/Refinement。

整个方案遵循以下原则：

1. `arm_1` 和 `arm_2` 始终拥有独立时间线。
2. 两臂在同一轮中可以从不同时间点开始输出动作。
3. 下一轮视频入口由当前片段末尾尚不稳定的动作边界决定。
4. 已经稳定的历史动作保留；可能被下一轮重新观察的尾部动作会被替换。
5. 中间片段不生成 `merged_timeline`。
6. 开启最终 merge 时，只有完整 episode 的所有片段预测结束后，才对两条完整机械臂时间线取边界并集。
7. 所有内部动作边界必须对应实际发送给 VLM 的采样图像。
8. 所有动作区间统一采用左闭右开语义 `[start_time, end_time)`。

## 2. 三种时间范围

递归标注中必须区分以下三个概念。

### 2.1 Episode 全局时间轴

episode 的时间轴固定为：

```text
[0, episode_duration)
```

所有片段、所有 stage、所有动作输出都使用这个全局时间轴。片段开始后不得重新从 0 计时。

### 2.2 当前视频窗口

第 `k` 轮发送给 VLM 的视频范围为：

```text
[video_start_k, video_end_k)
```

其中：

```text
video_end_k = min(video_start_k + 15.0, episode_duration)
```

这个范围决定当前轮实际发送哪些图像，但不代表两只机械臂都必须从 `video_start_k` 开始输出动作。

### 2.3 逐臂标注区间

当前轮分别为两臂维护：

```text
arm_1: [arm_1_annotation_start, video_end_k)
arm_2: [arm_2_annotation_start, video_end_k)
```

可能出现：

```text
当前视频窗口: [11.00, 26.00)
arm_1 标注区间: [11.00, 26.00)
arm_2 标注区间: [13.00, 26.00)
```

此时 `arm_2` 仍然会看到 `[11.00, 13.00)` 的图像，但这些图像只用于观察边界上下文。`arm_2` 不得为该上下文区间输出 action。

## 3. Episode 级帧缓存

启用递归模式后，框架先按各 stage 配置的 FPS 为完整 episode 建立图像序列缓存。

当前配置中：

```yaml
analysis.video.fps: 4.0
refinement.video.fps: 4.0
```

缓存 manifest 为每个采样点保存：

```json
{
  "frame_index": 390,
  "timestamp_sec": 13.0,
  "path": "sources/.../frame_000000390.jpg"
}
```

每张发送给 VLM 的图像前都有三类标签：

```text
<t=13.00s> <frame_index=390> <observation.images.image_0>
```

含义分别是：

- `<t=...s>`：原 episode 的全局时间。
- `<frame_index=...>`：原视频真实帧号。
- `<view_name>`：当前图像所属视角。

`frame_index` 必须从缓存 manifest 的真实 `timestamp_sec/frame_index` 对中读取，不允许使用 `time * fps` 反算。

## 4. 每轮三阶段处理

### 4.1 Scene

Scene 观看当前完整视频窗口，输出当前片段中的执行器、物体和候选交互物体。

双臂场景中执行器固定为：

```text
arm_1: 左机械臂
arm_2: 右机械臂
```

即使某只机械臂在当前窗口不可见，Scene 的执行器数量仍应与 `robot_type=bimanual` 匹配。

### 4.2 Analysis

Analysis 为每只机械臂分别输出一条动作时间线：

```json
{
  "executor_timelines": [
    {
      "executor": "arm_1",
      "actions": [
        {
          "start_evidence": "...",
          "start_time_hint": 11.0,
          "process_evidence": "...",
          "end_evidence": "...",
          "end_time_hint": 13.5,
          "action": "移动",
          "object": null,
          "target": null,
          "direction": null,
          "relation_evidence": null,
          "related_executors": [],
          "executor_relation": "无"
        }
      ]
    },
    {
      "executor": "arm_2",
      "actions": []
    }
  ],
  "uncertainties": []
}
```

约束如下：

1. 每条 action 只属于当前 `executor`。
2. 每个 action 的 `action` 必须是动作词表中的单个原子动作，不得组合多个动作词。
3. 静止区间也必须输出动作，例如“等待”“悬停”“保持”“夹持”。
4. 同一机械臂的 actions 必须无间隙、无重叠地覆盖该机械臂当前轮的逐臂标注区间。
5. 第一条 action 的 `start_time_hint` 必须等于该机械臂标注契约的起点。
6. 最后一条 action 的 `end_time_hint` 必须等于当前视频窗口右端点。
7. 相邻 action 共享同一内部边界。
8. 除最后一个结束边界外，所有边界必须来自输入图像的 `<t=...s>` 标签。
9. 两臂动作可以重叠，协作关系通过 `related_executors`、`executor_relation` 和 `relation_evidence` 表达，不合并成联合 action。

### 4.3 Refinement

Refinement 读取 Analysis 的原始 `executor_timelines`，重新观看当前片段并精修边界。

Refinement 必须：

1. 保持 executor 顺序不变。
2. 保持每个 executor 的 action 数量和顺序不变。
3. 不新增、删除、拆分或合并 action。
4. 原样保留 `action/object/target/direction` 和双臂关系字段。
5. 只将 `start_time_hint/end_time_hint` 精修为 `start_time/end_time`，并输出边界证据。
6. 分别服从 `arm_1` 和 `arm_2` 的逐臂标注契约。

片段级 Refinement 输出结构为：

```json
{
  "executor_timelines": [
    {
      "executor": "arm_1",
      "actions": [
        {
          "start_time": 11.0,
          "end_time": 13.5,
          "start_boundary_evidence": "...",
          "end_boundary_evidence": "...",
          "action": "移动",
          "object": null,
          "target": null,
          "direction": null,
          "relation_evidence": null,
          "related_executors": [],
          "executor_relation": "无"
        }
      ]
    }
  ],
  "uncertainties": []
}
```

片段级 Refinement 输出不包含：

```text
subtask_id
merged_timeline
```

这些字段只在完整 episode 处理结束后生成。

## 5. 逐臂标注契约

Analysis 和 Refinement 的 user prompt 都会收到两个完整占位符：

```text
{{ ctx.segment.arm_1_annotation_contract }}
{{ ctx.segment.arm_2_annotation_contract }}
```

实际渲染示例：

```text
arm_2：
- 标注起始采样帧：<t=13.00s> <frame_index=390> <observation.images.image_0>。
- 该机械臂必须输出的全视频时间区间为 [13.00s, 26.000000s)。
- 第一个 action 的开始时间必须为 13.00，并对应上述 frame_index=390 的图像。
- 最后一个 action 的结束时间必须为 26.000000。
- 视频窗开头 [11.000000s, 13.00s) 仅作为该机械臂的边界上下文，不得输出该区间内的 action。
```

这保证模型明确知道：

- 当前看到的视频从哪里开始。
- 当前机械臂从哪张真实采样帧开始标注。
- 哪些开头图像只是上下文。
- 最后一条 action 应结束在哪里。

## 6. 片段输出的确定性校验

每轮 Refinement 完成后，调度器先校验结果，再决定下一片段。

每个 executor 必须满足：

1. timeline 存在且 `actions` 非空。
2. 第一条 action 从该 executor 的标注起点开始。
3. 最后一条 action 在当前视频窗口右端点结束。
4. 相邻 actions 首尾相接。
5. actions 不得重叠。
6. 内部边界必须能够匹配 Refinement 实际发送图像的采样时间戳。
7. 模型输出时间允许在配置容差内匹配采样点，匹配后统一规范为缓存中的真实时间戳。
8. 当前窗口右端点允许作为最后一个 action 的结束边界，即使该右端点没有作为图像发送。

默认时间匹配容差为：

```yaml
timestamp_tolerance_sec: 0.011
```

校验失败时当前片段进入 retry；超过最大重试次数后停止该 episode，避免将非法边界写入全局时间线。

## 7. 下一片段入口选择规则

设当前视频窗口为：

```text
[S, E)
```

设 Refinement 输出中：

```text
n1 = arm_1 当前轮 action 数量
n2 = arm_2 当前轮 action 数量
l1 = arm_1 最后一条 action 的 start_time
l2 = arm_2 最后一条 action 的 start_time
```

如果 `E >= episode_duration`，说明已经到达 episode 末尾，不再创建下一片段，直接进入最终拼接。

否则按以下规则选择下一入口。

### 7.1 两臂都有多个 subtask

条件：

```text
n1 > 1 and n2 > 1
```

下一视频窗口起点：

```text
next_video_start = min(l1, l2)
```

下一轮逐臂标注起点：

```text
arm_1_annotation_start = l1
arm_2_annotation_start = l2
```

示例：

```text
当前窗口: [0, 15)
arm_1 最后 action: [11, 15)
arm_2 最后 action: [13, 15)

下一视频窗口: [11, 26)
arm_1 标注区间: [11, 26)
arm_2 标注区间: [13, 26)
```

`arm_2` 会看到 `[11, 13)`，但该区间只用于上下文。

### 7.2 只有一臂包含多个 subtask

条件示例：

```text
n1 = 1 and n2 > 1
```

只有一个 subtask 的机械臂通常输出了一个覆盖整个标注区间的长动作。该动作起点不能作为下一片段入口，否则会回到当前窗口起点，造成无进展循环。

因此忽略单 subtask 机械臂，仅使用另一臂最后一个 subtask 的起点：

```text
next_video_start = l2
arm_1_annotation_start = l2
arm_2_annotation_start = l2
```

反向情况 `n1 > 1 and n2 = 1` 同理。

示例：

```text
当前窗口: [0, 15)
arm_1: 只有一个 action [0, 15)
arm_2: 最后一个 action [9, 15)

下一视频窗口: [9, 24)
arm_1 标注区间: [9, 24)
arm_2 标注区间: [9, 24)
```

### 7.3 两臂都只有一个 subtask

条件：

```text
n1 = 1 and n2 = 1
```

两条 timeline 都没有提供可用于回退的内部语义边界。此时不保留固定重叠区间，直接从当前视频窗口结尾继续：

```text
next_video_start = E
arm_1_annotation_start = E
arm_2_annotation_start = E
```

示例：

```text
当前窗口: [0, 15)
下一视频窗口: [15, 30)
```

### 7.4 最小推进量与固定重叠兜底

如果至少一臂有多个 subtask，但按上述规则得到的候选入口距离当前 `S` 过近：

```text
next_video_start - S < min_advance_sec
```

则使用固定重叠兜底：

```text
fallback_target = E - fallback_overlap_sec
```

默认配置为：

```yaml
min_advance_sec: 0.25
fallback_overlap_sec: 2.0
```

`fallback_target` 会对齐到当前实际采样时间戳。兜底后：

```text
arm_1_annotation_start = next_video_start
arm_2_annotation_start = next_video_start
```

如果连固定重叠入口也无法严格向前推进，则停止并报告错误，避免无限循环。

两臂都只有一个 subtask 时不走固定重叠兜底，直接采用当前窗口结尾 `E`。

## 8. 片段之间的逐臂时间线拼接

系统不会简单地把每个片段的 actions 直接追加到全局数组，因为相邻片段包含重叠区域，尾部动作会被重新预测。

对每个 executor 独立执行尾段替换。

设：

```text
B = 当前轮该 executor 的 annotation_start
old_actions = 已累计的该 executor 全局 actions
new_actions = 当前轮该 executor 的 Refinement actions
```

拼接规则如下。

### 8.1 保留稳定历史

保留所有满足以下条件的旧 action：

```text
old_action.end_time <= B
```

这些 action 完全位于本轮重标区域之前，不会再被修改。

### 8.2 处理被兜底边界切开的旧长动作

如果固定重叠或单 subtask 规则使 `B` 落在某条旧 action 内部：

```text
old_action.start_time < B < old_action.end_time
```

则保留该旧 action 的前缀，并把结束时间截断为 `B`：

```text
[old_start, old_end) -> [old_start, B)
```

这一步保证即使入口来自固定重叠点，`[0, B)` 的历史时间线也不会丢失。

### 8.3 丢弃旧的临时尾段

旧时间线中从 `B` 开始或覆盖 `B` 之后的部分视为 provisional tail，由当前轮重新预测结果替换。

### 8.4 追加当前轮结果

要求：

```text
new_actions[0].start_time == B
```

然后生成：

```text
stitched_actions = retained_old_actions + new_actions
```

`arm_1` 和 `arm_2` 使用各自的 `B` 独立执行该过程。

### 8.5 示例

上一轮累计结果：

```text
arm_1:
  [0, 5) 等待
  [5, 11) 接近
  [11, 15) 移动    <- provisional tail

arm_2:
  [0, 8) 等待
  [8, 13) 夹持
  [13, 15) 保持    <- provisional tail
```

下一轮标注起点：

```text
arm_1 B=11
arm_2 B=13
```

下一轮重新预测：

```text
arm_1 new: [11, 18), [18, 26)
arm_2 new: [13, 20), [20, 26)
```

拼接后：

```text
arm_1:
  [0, 5), [5, 11), [11, 18), [18, 26)

arm_2:
  [0, 8), [8, 13), [13, 20), [20, 26)
```

旧的 `[11, 15)` 和 `[13, 15)` 被新一轮结果替换，而更早的动作保持不变。

## 9. 中间结果与断点恢复

每个片段保存于：

```text
<episode_run_dir>/segments/segment_XXXX/
```

其中每个 stage 都保存自己的：

```text
output.json
raw_text.txt
system_prompt.txt
user_prompt.txt
video_meta.json
```

episode 根目录下的 `progress.json` 保存：

- 下一片段编号。
- 下一视频窗口起点。
- 两臂下一轮各自的 annotation start。
- 当前累计的两条 executor timeline。
- 已访问的视频入口。
- retry 次数和最后一次调度决策。
- 完成或失败状态。

中断恢复时从 `progress.json` 继续，不需要重新预测已经提交的历史片段。

## 10. 为什么中间片段不生成 merged_timeline

片段末尾 action 是临时结果，会在下一片段中从其起点重新标注。如果每个片段结束后立即生成 merged timeline，会出现：

1. 临时尾段被当成稳定 subtask。
2. 下一片段重标后，旧 merged 片段和新结果冲突。
3. subtask ID 在不同窗口中重复编号。
4. 无法保证最终 merged 边界来自完整的双臂时间线。

因此中间 Refinement 输出只保留逐臂 actions。只有 episode 到达最终结束边界后，才执行一次全局后处理。

最终 merge 由 `segment_loop.merge_after_episode` 控制：

- `true`（默认）：为完整逐臂时间线分配稳定 `subtask_id`，并生成 `merged_timeline`。
- `false`：仍完成全部片段预测、逐臂尾段替换和完整时间线拼接，也会分配稳定 `subtask_id`，但不计算边界并集，不在最终 `output.json` 中写入 `merged_timeline`。

这个开关只控制 episode 结束后的 merge，不改变三阶段预测、递归切分或逐臂时间线拼接。

## 11. 完整 episode 的最终处理

当最后一个视频窗口满足：

```text
video_end >= episode_duration
```

系统先将最后一轮 actions 拼接进两条累计时间线，然后执行以下步骤。

### 11.1 为完整时间线分配稳定 subtask ID

分别按照两臂完整 actions 的全局顺序生成：

```text
arm_1_subtask_0001
arm_1_subtask_0002
...

arm_2_subtask_0001
arm_2_subtask_0002
...
```

ID 此时只生成一次，因此不会受到片段局部编号影响。

### 11.2 收集双臂全部边界

从两条完整 timeline 收集所有：

```text
start_time
end_time
```

并去重排序：

```text
B = sorted(unique(all_arm_1_boundaries + all_arm_2_boundaries))
```

### 11.3 构造边界并集区间

对每一对相邻边界：

```text
[B[i], B[i+1])
```

分别查找覆盖该区间的 `arm_1` 和 `arm_2` subtask。

由于最终两条机械臂时间线都必须连续覆盖完整 episode，因此每个 merged 区间都必须能够找到两臂各自唯一的覆盖 subtask。缺失、重叠或多重覆盖都会触发错误，而不是生成不完整 merged 结果。

### 11.4 判断每只机械臂在 merged 区间中的阶段

对于某个 merged 区间 `[M_start, M_end)` 和覆盖它的 action `[A_start, A_end)`：

| 条件 | phase |
|---|---|
| `A_start == M_start` 且 `A_end == M_end` | `start_and_finish` |
| 只有 `A_start == M_start` | `start` |
| 只有 `A_end == M_end` | `continue_and_finish` |
| 两者都不满足 | `continue` |

### 11.5 生成自然语言描述

描述规则：

```text
start:
  arm_x开始执行...

continue:
  arm_x继续执行...

start_and_finish:
  arm_x开始执行...，持续X秒后结束该动作

continue_and_finish:
  arm_x继续执行...，X秒后结束该动作
```

同一 merged 区间中，开始新动作的机械臂描述优先，继续上一动作的机械臂描述随后。

## 12. 最终 output.json

最终结果保存到：

```text
<episode_run_dir>/output.json
```

当 `merge_after_episode: true` 时，结构为：

```json
{
  "executor_timelines": [
    {
      "executor": "arm_1",
      "actions": [
        {
          "subtask_id": "arm_1_subtask_0001",
          "start_time": 0.0,
          "end_time": 3.0,
          "action": "等待"
        }
      ]
    },
    {
      "executor": "arm_2",
      "actions": [
        {
          "subtask_id": "arm_2_subtask_0001",
          "start_time": 0.0,
          "end_time": 5.0,
          "action": "保持"
        }
      ]
    }
  ],
  "uncertainties": [],
  "merged_timeline": [
    {
      "segment_id": "segment_0001",
      "start_time": 0.0,
      "end_time": 3.0,
      "arm_1": {
        "subtask_id": "arm_1_subtask_0001",
        "action": "等待",
        "phase": "start_and_finish",
        "description": "arm_1开始执行等待，持续3秒后结束该动作"
      },
      "arm_2": {
        "subtask_id": "arm_2_subtask_0001",
        "action": "保持",
        "phase": "start",
        "description": "arm_2开始执行保持"
      },
      "segment_description": "arm_1开始执行等待，持续3秒后结束该动作；arm_2开始执行保持。"
    }
  ]
}
```

最终 JSON 同时保留：

1. `arm_1` 完整时间线。
2. `arm_2` 完整时间线。
3. 所有片段累计的 `uncertainties`。
4. 当 `merge_after_episode: true` 时，基于两臂完整边界并集生成的 `merged_timeline`。

当 `merge_after_episode: false` 时，前三项保持不变，省略整个 `merged_timeline` 字段。

## 13. 最终结果必须满足的不变量

一个成功完成的 episode 必须满足：

### 13.1 单臂时间线

对 `arm_1` 和 `arm_2` 分别满足：

```text
first_action.start_time == 0
last_action.end_time == episode_duration
```

并且对任意相邻 action：

```text
previous.end_time == next.start_time
```

不得存在空洞或重叠。

### 13.2 Merged 时间线

仅当 `merge_after_episode: true` 时生成，并且必须满足：

```text
merged_timeline[0].start_time == 0
merged_timeline[-1].end_time == episode_duration
```

任意相邻 merged segment 必须首尾相接。

每个 merged segment 必须同时引用一个 `arm_1` subtask 和一个 `arm_2` subtask。

### 13.3 中间结果

任意：

```text
segments/segment_XXXX/stages/refinement/output.json
```

都不得包含 `merged_timeline`。开启最终 merge 时，`merged_timeline` 只存在于 episode 根目录的最终 `output.json`；关闭时任何输出都不包含该字段。

## 14. 当前默认配置

核心递归配置如下：

```yaml
segment_loop:
  enabled: true
  window_duration_sec: 15.0
  result_stage: refinement
  merge_after_episode: true
  executors:
    - arm_1
    - arm_2
  max_retries_per_segment: 2
  max_segments: 1000
  frame_cache:
    enabled: true
    jpeg_quality: 95
    cleanup_on_complete: false
  postprocessor:
    policy: bimanual_last_action_start
    start_time_field: start_time
    end_time_field: end_time
    min_advance_sec: 0.25
    fallback_overlap_sec: 2.0
    require_sampled_timestamps: true
    timestamp_tolerance_sec: 0.011
```

## 15. 流程总览

```text
完整 episode
  |
  |-- 按 stage FPS 一次性建立全局帧缓存
  |
  |-- segment 0: 视频窗 [0, 15)
  |     |-- Scene
  |     |-- Analysis: arm_1/arm_2 各自完整覆盖标注区间
  |     |-- Refinement: 精修两条时间线
  |     |-- 校验真实采样时间戳
  |     |-- 按单/多 subtask 规则计算下一入口
  |     `-- 按两臂各自 annotation start 替换临时尾段
  |
  |-- segment 1..N
  |     `-- 重复完整三阶段和逐臂尾段替换
  |
  |-- 最后一轮到达 episode_duration
  |     |-- 拼接最后一轮两臂 actions
  |     |-- 为两条完整时间线分配全局 subtask_id
  |     `-- merge_after_episode=true 时取边界并集并生成描述
  |
  `-- 保存最终 output.json
        |-- arm_1 完整时间线
        |-- arm_2 完整时间线
        |-- uncertainties
        `-- merge_after_episode=true 时生成 merged_timeline
```
