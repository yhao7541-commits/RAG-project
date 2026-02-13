---
name: spec-sync
description: 将 DEV_SPEC.md 同步并拆分为 specs/ 章节文件。运行 sync_spec.py 后，通过 SPEC_INDEX.md 快速定位规范内容。任何依赖规范的工作前都应先执行。
metadata:
  category: documentation
  triggers: "sync spec, update spec, 同步规范"
allowed-tools: Bash(python:*) Read
---

# 规范同步（Spec Sync）

这个技能用于把主规范 `DEV_SPEC.md` 同步到 `.github/skills/spec-sync/specs/` 目录，生成可按章节阅读的规范文件。

> 这是所有“基于规范开发”任务的前置步骤。

---

## 什么时候使用

- 当你修改了 `DEV_SPEC.md` 之后
- 当 `specs/` 目录缺失、损坏或内容过期
- 在单独调试某个依赖规范的技能前
- 作为 `dev-workflow` 的第 1 阶段（自动执行）

---

## 使用方式

### 在工作流里（自动）

当你触发 `dev-workflow`（例如“下一阶段”“继续开发”）时，会自动执行 spec-sync，一般不需要手动操作。

### 手动执行（仅边界场景）

```bash
# 常规同步
python .github/skills/spec-sync/sync_spec.py

# 强制重建（即使未检测到变更）
python .github/skills/spec-sync/sync_spec.py --force
```

---

## 同步脚本会做什么

1. 读取项目根目录下的 `DEV_SPEC.md`
2. 计算哈希，判断是否发生变更
3. 按章节拆分写入 `specs/`
4. 生成导航索引 `SPEC_INDEX.md`

---

## 同步后如何阅读

1. 先读索引：

```text
.github/skills/spec-sync/SPEC_INDEX.md
```

2. 再按需要读对应章节，例如：

```text
.github/skills/spec-sync/specs/05-architecture.md
```

---

## 目录结构

```text
.github/skills/spec-sync/
├── SKILL.md
├── SPEC_INDEX.md
├── sync_spec.py
├── .spec_hash
└── specs/
    ├── 01-overview.md
    ├── 02-features.md
    ├── 03-tech-stack.md
    ├── 04-testing.md
    ├── 05-architecture.md
    ├── 06-schedule.md
    └── 07-future.md
```

---

## 重要规则

1. 不要直接编辑 `specs/` 下文件（这些是自动生成的）。
2. 只编辑 `DEV_SPEC.md`，然后执行同步脚本。
3. 若怀疑索引或章节异常，使用 `--force` 重建。
