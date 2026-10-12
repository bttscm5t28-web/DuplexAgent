# 快脑基准标注规范

## 委派数据 `bench_data/fast/delegation*.jsonl`

每行一个 JSON：

```json
{"id": "chat-001", "cat": "chitchat", "text": "用户这一轮说的话",
 "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "...", "interrupted": true}],
 "tasks": [{"id": "t1", "query": "后台任务描述", "status": "running", "progress": ["已找到3个结果"]}],
 "delegate": false, "action": "say", "forbid": ["1081"]}
```

- `delegate`（必填，兼容 `python -m duplexagent bench fast`）：这一轮是否应当产生 **新的** `Delegate`。
- `action`：期望的主动作，`say`（自己答）| `delegate` | `cancel`（取消正在运行的任务）| `amend`（给正在运行的任务补充信息）| `status`（回答任务进度，不新建任务）。
- `tasks`：本轮开始时的后台任务（`running` / `done`），用于进度、取消、补充、打断后续接等情形。
- `forbid`：快脑这一轮说出的话里 **不应该出现** 的内容（多用于“不要对已委派/正在处理的问题瞎猜答案”）。
- `history` 里 `interrupted: true` 表示助手那句话被用户打断，只播出了这一部分。

### 判定原则：什么时候委派

**委派给慢脑**（`delegate: true`）：
1. 需要实时 / 外部信息：天气、新闻、股价、汇率、赛程、营业时间、路况、最新版本等。
2. 需要用户私有数据或执行动作：日程、邮件、航班订单、提醒、发消息、下单、设置闹钟、控制设备。
3. 需要精确的非平凡计算：多位数乘除、复利、换算链、统计，**心算一步能出的不算**（3 加 5、100 的一半）。
4. 多步推理 / 规划 / 比较 / 调研：旅行行程、方案对比、学习计划、选购建议（需要具体型号参数的）。
5. 长内容生成：邮件、文章、总结、长段翻译、代码。
6. 需要高准确度而快脑没把握的专业问题：具体法规条款、药物剂量、具体历史数据。

**快脑自己答**（`delegate: false`）：
1. 闲聊、寒暄、情绪陪伴、观点、玩笑、简单建议（“睡不着怎么办”可以给一两句常识建议）。
2. 稳定的常识：中国首都、水的沸点、一年几个月、简单词语解释、短词翻译（“苹果英文怎么说”）。
3. 心算一步能出的算术。
4. 对正在运行的任务问进度（`action: status`）、要求取消（`action: cancel`）、补充条件（`action: amend`）——都 **不新建** 任务。
5. 澄清 / 确认 / 附和后的接话、被打断后的续接（除非续接的内容本身就是新的复杂请求）。
6. 指令不完整时先追问（“帮我订个”→ 问订什么），不委派。

## 融合数据 `bench_data/fast/fusion.jsonl`

```json
{"id": "fuse-001", "history": [...], "deliver": [{"id": "t1", "query": "...", "status": "done", "result": "..."}],
 "must_mention_any": [["北京", "天气"]], "must_contain_any": [["晴"], ["25"]], "correction": false, "max_chars": 80}
```

快脑在 `trigger="task_result"` 时的输出要满足：
- **点明问题**：`must_mention_any` 每组至少命中一个词（说明这是哪个问题的结果）。
- **传达要点**：`must_contain_any` 每组至少命中一个词（结果里的关键事实）。
- **简短口语**：不超过 `max_chars` 个字；不含 Markdown / 列表符号 / 表情。
- **冲突更正**：`correction: true` 时，之前助手说过与结果冲突的话，回复须明确更正（含“更正 / 不对 / 说错 / 不准 / 纠正 / 其实 / 应该是”之一）；`false` 时不应出现更正用语。
- 任务失败（`status: failed`）时要说明没完成，不能编结果。
