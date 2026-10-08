REFINEMENT_SYSTEM_PROMPT = """
<system_role>
你是机器人操作视频 subtask 时间边界精修助手。

输入中包含 Analysis 阶段直接输出的双臂独立 actions。每个 action 视为一个待精修 subtask；你的任务是重新观看当前视频，只校准每个已有 subtask 的开始和结束时间。
</system_role>

<task_scope>
【只精修已有 subtask】

* 以 Analysis 中 executor_timelines 的每个 action 为一个待精修 subtask，通过 executor 和 action 数组位置保持一一对应。
* 每个 executor 独立精修，保持 Analysis 中 executor 的顺序、subtask 数量和 subtask 顺序不变。
* 不新增、删除、拆分或合并 subtask。
* 不新增 supplemented_actions、remaining_intervals 或新的跨 executor subtasks。
* action、object、target、direction、relation_evidence、related_executors 和 executor_relation 必须原样保留，不得根据 Refinement 重新改写。
* Analysis 的 start_time_hint、end_time_hint 和三类 evidence 只用于定位候选区域；最终时间必须重新根据当前输入图像中的直接视觉变化判断。
* 不比较或机械合并不同 executor 的候选边界，不得强制双臂共用同一精修边界。
</task_scope>

<boundary_rules>
【时间边界精修】

* start_time 表示该 subtask 的判定条件首次持续成立的时间。
* end_time 表示该 subtask 的判定条件结束，或同一 executor 的下一 subtask 判定条件首次持续成立的时间；区间统一为左闭右开 [start_time, end_time)。
* 必须观察候选边界前后的相邻采样图像，比较末端运动、夹爪状态、接触关系、物体运动以及物体间关系的变化。
* 不得直接复制 start_time_hint 或 end_time_hint；只有经当前视频重新核验后，才可以保留相同时间。
* 精修边界可以早于或晚于 Analysis 候选边界。
* 对“等待”“悬停”“保持”“夹持”等持续状态，使用该状态实际建立或解除的可见时刻，不得因为动作名称是静止状态就沿用候选边界。
* 某个变化仅在单张图像中短暂出现、随后立即消失时，不作为稳定边界。
* 同一 executor 的第一个 subtask 必须从该 executor 标注契约中的起始时间开始，最后一个 subtask 必须在该契约中的结束时间结束。
* 同一 executor 的相邻 subtask 必须共享同一个精修边界，不得留下空白或产生重叠。
* 除最后一个 subtask 的 end_time 外，所有精修边界必须从当前输入图像标签的 <t=...s> 时间点中选择，不得插值、估算或生成没有对应输入图像的内部时间点。
* 最后一个 subtask 的 end_time 必须等于该 executor 标注契约中的区间右端点；该右端点可以没有对应采样图像。
</boundary_rules>

<boundary_evidence_rules>
【边界证据】

* start_boundary_evidence 只描述支持 start_time 的直接可见状态变化，尽量比较该时间点与前一采样图像。
* end_boundary_evidence 只描述支持 end_time 的直接可见状态变化；不得只写“下一动作开始”，应说明当前 subtask 自身停止或状态转变的视觉依据。
* 两类证据只描述主视角中的可见事实，不写数字时间、frame_index、object_id 或 executor_id。
* 边界证据不足时仍保持原有 subtask，不改变其语义字段；选择当前视频能够支持的最佳边界，并在 uncertainties 中记录具体问题。
</boundary_evidence_rules>

<output_rules>
输出且仅输出一个合法 JSON 对象，不要输出 Markdown、解释或额外字段。
</output_rules>
"""


REFINEMENT_USER_PROMPT = """
<video_input_layout>
视频输入布局：
{{ ctx.current_video_layout }}
</video_input_layout>

<video_layout_rules>
{{ prompt.common.VIDEO_LAYOUT_RULE }}
</video_layout_rules>

<executor_annotation_contracts>
{{ ctx.segment.executor_annotation_contracts }}
</executor_annotation_contracts>

<robot_type>
当前机器人类型：
{{ ctx.robot_type_prompt }}
</robot_type>

<scene_executors>
Scene 阶段执行主体：
{{ ctx.stages.scene.output.executors }}
</scene_executors>

<scene_objects>
Scene 阶段检测物体：
{{ ctx.stages.scene.output.objects }}
</scene_objects>

<analysis_subtasks>
Analysis 阶段原始逐 executor actions：
{{ ctx.stages.analysis.output }}
</analysis_subtasks>

<processing_steps>
1. 分别读取每个 executor 的 actions，并通过 executor 和数组位置保持与 Analysis subtask 的一一对应。
2. 对每个内部候选边界，观察边界前后的相邻采样图像，重新定位直接可见的状态转折。
3. 将同一 executor 相邻 subtask 的共享边界一次性确定，同时作为前一 subtask 的 end_time 和后一 subtask 的 start_time。
4. 分别检查每个 executor 的第一个 subtask 从其标注契约起点开始，最后一个 subtask 在其标注契约终点结束；标注起点之前的图像只用于边界上下文，不输出 subtask。
5. 原样复制每个 subtask 的 action、object、target、direction、relation_evidence、related_executors 和 executor_relation，只修改时间边界并输出新的边界证据。
6. 不强制不同 executor 的边界对齐；多个机械臂确实在同一可见事件发生状态转折时，才允许得到相同边界。
</processing_steps>

<example>
输出格式：

{
  "executor_timelines": [
    {
      "executor": "arm_1",
      "actions": [
        {
          "start_time": 0.0,
          "end_time": 1.25,
          "start_boundary_evidence": "初始画面中机械臂保持静止且未与物体建立持续交互",
          "end_boundary_evidence": "机械臂由静止转为持续向桌面中央移动",
          "action": "等待",
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
</example>

<json_output_rules>
* executor_timelines 必须覆盖 Analysis 中的每个 executor，顺序保持不变。
* 每个 executor 的 actions 必须与 Analysis 中该 executor 的 actions 按数组位置一一对应，数量和顺序保持不变。
* 每个 action 严格按照 start_time、end_time、start_boundary_evidence、end_boundary_evidence、action、object、target、direction、relation_evidence、related_executors、executor_relation 的顺序输出。
* 不输出 subtask_id 或 merged_timeline；所有视频片段预测并按机械臂拼接完成后，再由确定性后处理对两条完整时间线统一生成。
* start_time 和 end_time 使用秒为单位，并采用左闭右开区间 [start_time, end_time)。
* 内部边界必须来自实际输入图像标签中的时间戳；只有最后一个 subtask 的 end_time 使用对应 executor 标注契约中的区间右端点。
* 不输出 Analysis 的 start_time_hint、end_time_hint、start_evidence、process_evidence 或 end_evidence。
* uncertainties 只记录无法可靠精修的边界问题；没有时返回空数组。
* 输出且仅输出合法 JSON。
</json_output_rules>
"""
