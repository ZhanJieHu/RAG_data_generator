# Huishang QCFA Agent
（本项目以徽商研究为例，其数据合成思想可以用于各类专业领域。）
徽商研究数据集生成工具 —— 从公开书籍文本中自动抽取经营活动事实，生成 QCFA（Question-Chunk-Fact-Answer）结构化问答数据。

## 项目背景

面向徽商历史研究构建 RAG（检索增强生成）或领域问答系统时，最大的难点往往不是模型本身，而是缺少稳定、规整、可追溯的数据资产。徽商研究材料多来自古籍、地方志、学术著作和历史文献，文本体量大、结构松散，直接用于检索、评估或微调时成本很高。

本项目尝试把大模型能力落到数据生产流程中：先将长文本切分为可处理的 Chunk，再从 Chunk 中抽取经营活动相关 Fact，最后生成带有原文依据的 Question 与 Answer。整个过程将非结构化文本转化为可检索、可评估、可复用的 JSONL 数据集。

## 项目目标

本项目的核心目标是从徽商相关书籍中，自动构建结构化、可追溯的中文问答数据集：

```text
书籍原文 -> 文档分块 -> 事实抽取 -> QCFA 生成
```

每条 QCFA 数据都保留问题、答案、事实和原文片段，既能用于 RAG 系统评估，也能作为领域知识库或后续模型训练的数据基础。

## 项目优势

- **面向 Data 的 AI 流水线**：不是简单调用 LLM 生成文本，而是围绕数据生产、数据结构、数据追溯和数据复用来组织全流程。
- **可追溯**：每个问题和答案都能回到原始 Chunk 与抽取出的 Fact，便于人工复核和质量控制。
- **结构化输出**：统一生成 JSONL，方便后续接入检索系统、评测脚本、标注平台或训练流程。
- **低耦合可扩展**：分块、事实抽取、问答生成三个阶段相互独立，可替换模型、调整 Prompt，也可插入人工审核环节。
- **领域适配**：Prompt 聚焦徽商经营活动事实，强调基于原文，不引入外部知识，降低幻觉对数据集的污染。

## Pipeline 概览

```text
data/books/
  └── 商帮书籍1.txt
          |
          v
  +-- Phase 1: 文档分块 ------------------+
  |  python main.py --mode chunk          |
  |  按字符数将书籍切分为等长 Chunk        |
  +----------------+----------------------+
                   |
                   v
  outputs/huishang_qcfa/chunks.jsonl
                   |
                   v
  +-- Phase 2: 事实抽取 ------------------+
  |  python main.py --mode extract-facts  |
  |  调用 LLM 从 Chunk 提取经营活动事实    |
  +----------------+----------------------+
                   |
                   v
  outputs/huishang_qcfa/fact_candidates.jsonl
                   |
                   v
  +-- Phase 3: QCFA 生成 -----------------+
  |  python main.py --mode generate-qcfa  |
  |  从 Fact 生成 Question 与 Answer       |
  +----------------+----------------------+
                   |
                   v
  outputs/huishang_qcfa/qcfa_candidates.jsonl
```

## 项目结构

```text
huishang-qcfa-agent/
├── main.py                        # Pipeline 入口，支持三种 mode
├── settings.yaml                  # 配置：分块字符数、API Key、模型名等
├── data/
│   └── books/
│       └── 商帮书籍1.txt           # 当前示例输入书籍
├── src/
│   ├── document_processor.py      # Phase 1: 文本分块
│   ├── agents/
│   │   ├── fact_extraction_agent.py   # Phase 2: 事实抽取
│   │   └── qcfa_generation_agent.py   # Phase 3: QCFA 生成
│   └── prompts/
│       ├── fact_extraction.md     # 事实抽取 Prompt
│       └── qcfa_generation.md     # QCFA 生成 Prompt
├── outputs/
│   └── huishang_qcfa/
│       ├── chunks.jsonl           # Phase 1 输出：分块结果
│       ├── fact_candidates.jsonl  # Phase 2 输出：候选事实
│       └── qcfa_candidates.jsonl  # Phase 3 输出：QCFA 问答对
└── tests/
    ├── test_chunk.py
    ├── test_fact_extraction.py
    └── test_qcfa_generation.py
```

## 安装

```bash
pip install pyyaml openai
```

## 使用方法

### 1. 配置

编辑 `settings.yaml`：

```yaml
top_char: 1200

deepseek:
  api_key: "sk-你的API密钥"
  model: "deepseek-v4-flash"
  base_url: "https://api.deepseek.com"
```

### 2. 放置书籍

将 `.txt` 或 `.md` 格式的徽商相关书籍放入 `data/books/` 目录。当前示例输入文件为：

```text
data/books/商帮书籍1.txt
```

### 3. 运行 Pipeline

```bash
# Phase 1: 文档分块
python main.py --mode chunk

# Phase 2: 事实抽取，默认读取 outputs/huishang_qcfa/chunks.jsonl
python main.py --mode extract-facts

# Phase 3: QCFA 生成，默认读取 outputs/huishang_qcfa/fact_candidates.jsonl
python main.py --mode generate-qcfa
```

默认输入 / 输出路径可省略，也可显式指定：

```bash
python main.py --mode extract-facts \
    --input outputs/huishang_qcfa/chunks.jsonl \
    --output outputs/huishang_qcfa/fact_candidates.jsonl
```

主要输出文件位于：

```text
outputs/huishang_qcfa/chunks.jsonl
outputs/huishang_qcfa/fact_candidates.jsonl
outputs/huishang_qcfa/qcfa_candidates.jsonl
```

### 4. 运行测试

```bash
python -m pytest tests/ -v
```

## 数据格式

### Phase 1: ChunkRecord

```json
{"text": "文段内容...", "char_count": 1200}
```

### Phase 2: FactRecord

```json
{
  "chunk": "包含事实的原始文段",
  "fact": "核心事实陈述。解释：原文依据说明。"
}
```

### Phase 3: QCFARecord

```json
{
  "question": "徽商中的多数人主要通过什么商业活动发家？",
  "answer": "主要通过长途贩运活动发家。",
  "chunk": "包含该事实的原始文段",
  "fact": "徽商中的多数人通过长途贩运活动发家。解释：..."
}
```

## 结果展示

最终生成的 QCFA 数据保存在：

```text
outputs/huishang_qcfa/qcfa_candidates.jsonl
```

该文件为 JSONL 格式，每一行都是一条独立的 QCFA 记录。下面示例来自该文件中的真实记录；为便于 README 阅读，`chunk` 字段只保留了关键原文片段：

```json
{
  "question": "明嘉靖时，歙县人吴柯弃儒服贾后取得了怎样的财富结果？",
  "answer": "以经商获得“十致千金”的财富。",
  "chunk": "明嘉靖时，歙人吴柯认为“士而成功也十一，贾而成功也十九”,遂弃儒服贾，果然“十致千金”...",
  "fact": "明嘉靖时，歙县人吴柯弃儒服贾，以经商获得“十致千金”的财富。解释：文段明确记载“明嘉靖时，歙人吴柯认为‘士而成功也十一，贾而成功也十九’，遂弃儒服贾，果然‘十致千金’”，这是一次具体的弃儒经商活动，产生了致富的结果。"
}
```

这条记录体现了 QCFA 的核心价值：`question` 可作为用户查询或评测题目，`answer` 给出标准答案，`fact` 提炼出结构化事实，`chunk` 则保留原文依据，便于追溯和复核。

## 技术栈

- **Python 3.10+**
- **DeepSeek API** — 提供 LLM 能力
- **OpenAI Python SDK** — 兼容 DeepSeek API 调用
- **JSONL** — 结构化数据存储格式
- **PyYAML** — 配置文件管理

## 数据来源说明

本文档展示用的数据来自互联网公开书籍。
