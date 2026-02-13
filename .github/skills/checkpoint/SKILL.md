---
name: checkpoint
description: 在任务实现与测试完成后，生成工作总结、更新 DEV_SPEC.md 进度，并可选准备/执行 git commit。属于 dev-workflow 最后阶段。
metadata:
  category: progress-tracking
  triggers: "checkpoint, save progress, 完成检查点, 保存进度, 任务完成"
allowed-tools: Bash(python:*) Bash(git:*) Read Write
---

# 检查点与进度持久化（Checkpoint）

此技能负责“收口”：把本轮完成内容写清楚、把进度落到 `DEV_SPEC.md`、并让用户决定是否提交 Git。

> 单一职责：总结 → 落表 → 准备下一轮

---

## 适用场景

- 当前任务实现与测试都已完成
- 需要手动同步 `DEV_SPEC.md` 进度
- 需要生成提交信息（可选执行 commit）

---

## 标准流程

1. Step 1：生成工作总结
2. Step 1.5：用户确认总结是否准确（确认“做了什么”）
3. Step 2：更新 `DEV_SPEC.md`（自动执行）
4. Step 3：生成 commit 信息并询问是否提交（确认“要不要提交”）

---

## Step 1：工作总结

收集并输出：

- 任务 ID / 名称
- 新增与修改文件
- 测试结果（通过/失败、可选覆盖率）
- 迭代次数（修复-复测轮次）
- 对应规范章节

输出示例：

```text
TASK COMPLETED: [Task ID] [Task Name]
Files Changed:
  Created: ...
  Modified: ...
Test Results: ...
Iterations: N
Spec Reference: DEV_SPEC.md Section X.Y
```

---

## Step 1.5：用户确认总结

必须等待用户明确回复后再继续：

- `confirm/确认`：进入 Step 2
- `revise/修改`：按反馈重生总结

> 注意：这是确认“总结内容是否准确”，不是确认是否 commit。

---

## Step 2：写入进度（自动）

### 2.1 更新单任务状态

在 `DEV_SPEC.md` 中定位任务并更新状态：

- `[ ]` → `[x]`
- `(进行中)` → `(已完成)`
- 或匹配文档既有标记样式做等价更新

### 2.2 更新总体进度表（必须）

更新“阶段完成数”和“总计完成数”，并重算百分比：

`进度 = 已完成 / 总任务数`

### 2.3 输出结果

```text
✅ DEV_SPEC.md Progress Updated
Task: [ID] [Name]
Status: [ ] -> [x]
Phase Progress: updated
```

---

## Step 3：提交准备（可选）

### 3.1 生成提交信息

建议格式：

```text
<type>(<scope>): [Phase X.Y] <brief description>
```

类型建议：

- `feat` 新功能
- `fix` 缺陷修复
- `refactor` 重构
- `test` 测试相关
- `docs` 文档相关
- `chore` 配置/杂项

### 3.2 询问是否执行 commit

- `yes/commit/是`：执行 `git add` + `git commit`
- `no/skip/否`：结束流程，用户后续手动提交

---

## 快捷指令

| 用户说法 | 行为 |
|---|---|
| `checkpoint` / `完成检查点` | 跑完整流程（含两次确认） |
| `save progress` / `保存进度` | 执行 Step 1.5 + Step 2 |
| `commit message` / `生成提交信息` | 仅 Step 3.1 |
| `commit for me` / `帮我提交` | Step 3.1 + 3.2 + 执行提交 |

---

## 硬性规则

1. `DEV_SPEC.md` 是全局进度唯一事实源。
2. 保持原有文档格式风格，不做无关重排。
3. 一次只更新一个任务，不批量改状态。
4. 两次确认都不能省略：
   - 确认总结
   - 确认是否提交
5. 更新单任务后，必须同步更新总体进度表。
