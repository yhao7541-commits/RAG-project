---
name: implement
description: 按规范驱动完成实现。先读规范、抽取设计原则、制定文件策略，再写可运行代码并自检。用于“实现功能/写代码/构建模块”等场景。
metadata:
  category: implementation
  triggers: "implement, write code, build module, 实现, 写代码"
allowed-tools: Read Write Bash(python:*) Bash(pytest:*)
---

# 实现阶段标准作业（Implement from Spec）

你是模块实现负责人。用户要求“实现”时，必须遵循以下固定流程。

> 前置依赖：`spec-sync` 已完成，且可读取 `.github/skills/spec-sync/specs/`。

---

## Step 1：读取与分析规范

### 1.1 先索引，后定位

1. 先读 `.github/skills/spec-sync/SPEC_INDEX.md`
2. 再读目标章节，而不是通读整份 `DEV_SPEC.md`

### 1.2 抽取任务要点

- 输入/输出约束
- 依赖与外部库要求
- 必须修改/新增的文件
- 验收标准与测试方法

### 1.3 抽取设计原则（必须）

从 `06-schedule.md` + `03-tech-stack.md` / `05-architecture.md` 中提取当前任务适用原则：

- 可插拔（Pluggable）
- 工厂路由（Factory）
- 配置驱动（Config-Driven）
- 失败回退（Fallback）
- 幂等性（Idempotent）
- 可观测性（Observable）

输出模板：

```text
DESIGN PRINCIPLES FOR THIS TASK
Task: [ID] [Name]
1) [原则] - [落地要求]
2) [原则] - [落地要求]
Source: specs/xx-xxx.md Section x.x
```

---

## Step 2：技术计划

编码前先列清：

1. 文件策略：新增/修改哪些文件（与排期“修改文件”字段一致）
2. 接口策略：是否需要先定义抽象类或工厂签名
3. 配置策略：需要新增哪些 `settings.yaml` 字段
4. 依赖策略：是否需要更新依赖文件
5. 原则核对：Step 1.3 的每条原则都有落地路径

---

## Step 3：实现代码

### 编码标准

1. 全函数签名加类型标注（Type Hints）
2. 类与方法写清晰 Docstring
3. 禁止硬编码（改为配置或注入）
4. 函数短小、单一职责、命名可读
5. 先校验输入，失败快速报错（Fail Fast）

### 错误处理

- 外部依赖（LLM/DB/HTTP）必须有可读异常信息
- 异常文案要可定位：模块名 + 失败类型 + 关键信息

---

## Step 4：实现后自检（静态）

> 这里只做“代码自检”；测试执行由 testing-stage 负责。

检查项：

- [ ] 满足规范约束
- [ ] 设计原则有对应实现
- [ ] 测试文件结构完整（即便暂未执行）
- [ ] 无 `pass` 占位；若未完成，使用明确 `NotImplementedError`

输出模板：

```text
DESIGN PRINCIPLES APPLIED
[x] Pluggable: ...
[x] Factory: ...
[x] Config-Driven: ...
[x] Error Handling: ...
```

---

## 章节快速定位

- 架构问题：`05-architecture.md`
- 技术实现：`03-tech-stack.md`
- 测试要求：`04-testing.md`
- 排期进度：`06-schedule.md`
