---
name: testing-stage
description: 在实现完成后执行系统化测试。根据任务性质自动选择单测/集成/E2E，运行 pytest 并输出可执行反馈。属于 dev-workflow 第 4 阶段。
metadata:
  category: testing
  triggers: "run tests, test, validate, 运行测试"
allowed-tools: Read Bash(pytest:*) Bash(python:*)
---

# 测试阶段技能（Testing Stage）

你是质量保障工程师。实现阶段完成后，必须先测试，再进入下一阶段。

> 前置条件：`implement` 已完成。

---

## 测试类型决策矩阵

根据任务性质选择测试类型（优先参考 `specs/06-schedule.md` 中“测试方法”字段）：

| 任务特征 | 推荐测试 | 原因 |
|---|---|---|
| 单模块、无外部依赖 | 单元测试 | 快速、隔离、稳定 |
| 仅工厂/接口定义 | 单元测试（mock/fake） | 验证路由逻辑即可 |
| 依赖真实 DB/文件系统 | 集成测试 | 需验证真实交互 |
| 多模块编排（pipeline） | 集成测试 | 需验证协同链路 |
| CLI/用户入口 | E2E | 验证完整用户路径 |

---

## 目标

1. 校验实现是否满足规范
2. 运行对应测试并记录结果
3. 发现问题时提供可执行修复建议

---

## Step 1：确定测试范围与类型

1. 读取实现阶段产物（改动文件列表）
2. 将源码映射到测试文件（unit/integration/e2e）
3. 读取 schedule 中“测试方法”并确认最终测试类型

输出模板：

```text
TEST SCOPE IDENTIFIED
Task: [ID] [Name]
Modified Files:
- src/...

Spec Test Method: pytest -q tests/...
Selected Test Type: Unit/Integration/E2E
Rationale: ...
```

---

## Step 2：执行测试

### 2.1 先确认测试文件存在

```bash
ls tests/unit/test_<module>.py
ls tests/integration/test_<module>.py
```

### 2.2 执行测试

```bash
pytest -v tests/unit/test_<module>.py
```

### 2.3 若测试缺失

输出 `MISSING_TESTS`，并回退到 implement 阶段补测。

---

## Step 3：分析结果

### 全部通过

- 返回 `PASS`
- 标记可进入下一阶段

### 存在失败

- 返回 `FAIL`
- 必须包含：失败用例、报错摘要、根因判断、修复建议

---

## Step 4：反馈迭代

1. 生成修复报告并回传 implement
2. implement 修复后重新测试
3. 最多迭代 3 次，超限升级用户决策

---

## 质量标准

- 单测应快速（建议 < 10 秒）
- 用例应确定性（避免随机失败）
- 单测不得调用真实外部 API
- 失败信息必须可操作

---

## 验证清单

- [ ] 新增公共方法有测试覆盖
- [ ] 测试命名清晰
- [ ] 测试目录放置正确
- [ ] Mock 策略符合测试类型
- [ ] 断言对齐规范验收标准
- [ ] 无硬编码路径或敏感信息
