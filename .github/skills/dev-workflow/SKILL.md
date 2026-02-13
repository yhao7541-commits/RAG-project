---
name: dev-workflow
description: 开发主编排技能。用于“下一阶段/继续开发/next task”等指令，按流水线依次调用 spec-sync、progress-tracker、implement、testing-stage、checkpoint，每轮只完成一个子任务。
metadata:
  category: orchestration
  triggers: "next task, proceed, continue development, 下一阶段, 继续开发"
allowed-tools: Read
---

# 开发工作流编排器（Development Workflow Orchestrator）

你是项目的流程编排者。用户要求“继续开发”时，必须按固定阶段顺序执行，不可跳步。

> 这是一个“元技能”：只负责流程衔接；各阶段细节请看对应技能文件。

---

## 阶段 0：激活虚拟环境（前置）

在执行任何阶段前，先激活项目虚拟环境：

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

---

## 流水线阶段

| 阶段 | 技能 | 作用 | 文件 |
|---|---|---|---|
| 1 | `spec-sync` | 同步规范文档 | `.github/skills/spec-sync/SKILL.md` |
| 2 | `progress-tracker` | 确认当前进度与下一任务 | `.github/skills/progress-tracker/SKILL.md` |
| 3 | `implement` | 按规范实现代码 | `.github/skills/implement/SKILL.md` |
| 4 | `testing-stage` | 运行并分析测试 | `.github/skills/testing-stage/SKILL.md` |
| 5 | `checkpoint` | 更新进度并收口 | `.github/skills/checkpoint/SKILL.md` |

---

## 标准流程

1. Stage 1：同步规范
2. Stage 2：校验进度并定位任务
3. 等待用户确认任务
4. Stage 3：实现
5. Stage 4：测试
6. Stage 5：检查点（更新 DEV_SPEC、可选提交）

### 测试失败处理

- 若 Stage 4 失败：回到 Stage 3 修复，再次测试
- 最多迭代 3 次；超过则升级给用户人工决策

---

## 阶段间上下文传递

| 来源 | 目标 | 传递内容 |
|---|---|---|
| Stage 2 | Stage 3 | 任务 ID、任务名、相关规范章节 |
| Stage 3 | Stage 4 | 本次修改文件、模块路径 |
| Stage 4(失败) | Stage 3 | 失败用例、报错摘要、修复建议 |
| Stage 4(通过) | Stage 5 | 测试结果、迭代次数 |

---

## 快捷指令

| 用户说法 | 行为 |
|---|---|
| `next task` / `下一阶段` | 跑完整流水线（1→5），完成一个子任务 |
| `continue` / `继续实现` | 只进入实现阶段（默认任务已知） |
| `status` / `检查进度` | 仅执行进度定位（Stage 2） |
| `sync spec` / `同步规范` | 仅执行 Stage 1 |
| `run tests` / `运行测试` | 仅执行 Stage 4 |

---

## 编排规则

1. **顺序优先**：默认按 1→2→3→4→5 执行。
2. **一次一个子任务**：每轮只推进一个任务，不批处理。
3. **规范为准**：进度与任务以 `DEV_SPEC.md` 为单一事实源。
4. **必须确认**：Stage 2 后需用户确认，再进入 Stage 3。
5. **先测后记**：Stage 4 通过后才能进入 Stage 5。
6. **两次确认**（Stage 5）：
   - 确认“本次完成内容”是否准确
   - 确认“是否执行 git commit”
