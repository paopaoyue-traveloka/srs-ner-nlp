# QueryNER 数据集说明

> 来源：[bltlab/queryner](https://github.com/bltlab/query-ner)  
> 论文：[QueryNER: Segmentation of E-commerce Queries (arXiv:2405.09507)](https://arxiv.org/abs/2405.09507)  
> HuggingFace：[bltlab/queryner](https://huggingface.co/datasets/bltlab/queryner)

---

## 数据集简介

QueryNER 是一个面向**电商搜索查询**的命名实体识别数据集。原始查询来自 Amazon ESCI（电商搜索意图）数据集，由 3 位标注者对查询进行 NER 标注，标注粒度为**词级别（token-level）**。

数据集规模：

| Split      | 样本数（查询条数） |
|------------|----------------|
| train      | ~8,000         |
| validation | ~1,000         |
| test       | ~1,200         |

---

## 数据集样例

### BIO CoNLL 格式（最终输出）

每行为 `词<TAB>标签`，空行分隔不同查询：

```
lego          B-creator
star          B-product_name
wars          I-product_name
set           B-core_product_type

cotton        B-material
waterproof    B-modifier
t-shirt       B-core_product_type
men           B-department

16            B-UoM
oz            I-UoM
protein       B-content
powder        B-core_product_type
used          B-condition
```

### JSONL offset 格式（仓库原始存储）

仓库不直接存储查询文本，仅存储标签偏移量（需与 Amazon ESCI 数据联合使用）：

```json
{"example_id": 903885, "labels": ["B-creator", "B-core_product_type", "I-core_product_type", "I-core_product_type"]}
{"example_id": 1682164, "labels": ["B-creator", "B-core_product_type", "I-core_product_type", "B-modifier", "I-modifier"]}
```

### HuggingFace 格式

通过 `load_dataset("bltlab/queryner")` 加载后，每条样本结构：

```python
{
    "tokens": ["lego", "star", "wars", "set"],
    "ner_tags": [3, 8, 9, 1]   # 整数索引，对应 label 列表
}
```

---

## NER 标签类型

共 **17 种实体类型**，采用 BIO 编码（B-/I- 前缀 + O），HuggingFace 版本共 35 个标签索引：

| 标签类型            | 说明                                      | 示例                              |
|---------------------|-------------------------------------------|-----------------------------------|
| `core_product_type` | 核心商品类型，查询的主体商品              | "shirt", "notebook", "headphones" |
| `creator`           | 品牌或创作者/作者                         | "Nike", "Harry Potter", "Apple"   |
| `product_name`      | 具体产品型号或系列名                      | "AirPods Pro", "Galaxy S23"       |
| `product_number`    | 产品编号/型号                             | "6s", "XR", "2080ti"              |
| `modifier`          | 通用修饰词（描述产品特征）                | "waterproof", "portable", "heavy duty" |
| `color`             | 颜色                                      | "red", "navy blue"                |
| `material`          | 材质/原料                                 | "cotton", "stainless steel"       |
| `department`        | 受众/部门（性别、年龄群体等）             | "women", "kids", "mens"           |
| `occasion`          | 使用场合或目的                            | "wedding", "camping", "office"    |
| `shape`             | 形状或外形                                | "round", "rectangular", "oval"    |
| `content`           | 内容/主题（适用于书籍、媒体等）           | "math", "Harry Potter", "cooking" |
| `UoM`               | 计量单位（Unit of Measure）               | "16 oz", "1 gallon", "pack of 2"  |
| `quantity`          | 数量                                      | "3 pack", "set of 4"              |
| `price`             | 价格相关词                                | "under 50", "cheap"               |
| `condition`         | 商品状态/成色                             | "used", "refurbished", "new"      |
| `time`              | 时间相关词                                | "2023", "vintage", "retro"        |
| `origin`            | 产地/来源地                               | "Japanese", "Italian", "organic"  |
| `O`                 | 不属于任何实体类型                        | 连词、介词等功能词                 |

训练集标签频率分布（共 28,457 个 token）：

| 标签                | 数量  |
|---------------------|-------|
| B-core_product_type | 6,681 |
| I-core_product_type | 4,096 |
| B-modifier          | 2,613 |
| B-creator           | 1,800 |
| I-content           | 1,602 |
| I-modifier          | 1,530 |
| B-department        | 1,346 |
| B-product_name      | 1,112 |
| B-content           | 1,060 |
| 其余标签            | 6,617 |

---

## 标注规范（BIO Scheme）

### BIO 编码规则

QueryNER 遵循标准 **BIO（Beginning-Inside-Outside）** 标注规范：

| 前缀 | 含义                                      |
|------|-------------------------------------------|
| `B-` | Begin，实体的**第一个** token              |
| `I-` | Inside，实体的**后续** token（非首个）     |
| `O`  | Outside，不属于任何实体                   |

### 核心规则

1. **每个实体的第一个词必须用 `B-` 标注**，即使前一个词是同类型实体，也必须另起一个 `B-`（不允许两个相邻实体合并为一个）。

2. **`I-` 标签只能紧跟同类型的 `B-` 或 `I-` 标签**，不能独立出现，也不能跟随不同类型的标签。

3. **单词实体**只用 `B-` 标注，不使用 `I-`。

4. **多个同类型实体相邻**时，每个实体单独用 `B-` 开始：
   ```
   harry    B-content
   potter   I-content
   hogwarts B-creator    # 新实体，重新用 B-
   ```

5. **标注以空格分词为基础**（whitespace tokenization），不做子词拆分。

### 标注示例解析

```
nike          B-creator          # 品牌名（单词实体）
air           B-product_name     # 产品名开始
force         I-product_name     # 产品名继续
1             I-product_name     # 产品名继续
running       B-modifier         # 修饰词（新实体）
shoes         B-core_product_type # 核心商品类型
men           B-department       # 受众

16            B-UoM              # 计量单位开始（16 oz）
oz            I-UoM
whey          B-content          # 内容词
protein       B-core_product_type
powder        I-core_product_type
used          B-condition        # 商品状态
```

### 与标准 CoNLL-2003 的异同

| 特性             | CoNLL-2003        | QueryNER               |
|------------------|-------------------|------------------------|
| 标注方案         | BIO / BIOES       | BIO                    |
| 实体类型数量     | 4（PER/LOC/ORG/MISC） | 16（电商专用）       |
| 文本类型         | 新闻文本          | 电商搜索查询           |
| 分词方式         | 标准分词          | 空格分词               |
| 句子长度         | 较长              | 极短（平均 3-5 词）    |
| 标注一致性检验   | 单人              | 3 位标注者，含 IAA 评估 |

---

## 快速开始

```bash
# 安装依赖
uv add datasets hanlp gliner2 wandb

# 列出所有已注册数据集
uv run main.py list

# 查看数据集统计 + 标签分布
uv run main.py stats queryner

# 查看样本和实体提取（默认 train split，5 条）
uv run main.py show queryner
uv run main.py show queryner --split test --n 3

# 微调训练（HanLP 后端）
uv run main.py train queryner --backend hanlp
uv run main.py train queryner --backend hanlp --epochs 30 --lr 2e-5 --best_metric f1 --early_stopping_patience 3

# 微调训练（GLiNER2 后端，支持 pretrained model）
uv run main.py train queryner --backend gliner2 --pretrained_model fastino/gliner2-base-v1
uv run main.py train queryner --backend gliner2 --pretrained_model fastino/gliner2-large-v1 --epochs 20 --batch_size 8 --early_stopping_patience 3

# 上传 WandB（复制 .env.example 为 .env 并填入 WANDB_API_KEY）
uv run main.py train queryner --backend hanlp --wandb_project ner-finetune --wandb_run hanlp_train_queryner
uv run main.py train queryner --backend gliner2 --wandb_project ner-finetune --wandb_run gliner2_train_queryner

# 独立评估已训练模型
uv run main.py validate queryner --backend hanlp --split test
uv run main.py validate queryner --backend hanlp --split test --model_dir .model/queryner/best
uv run main.py validate queryner --backend gliner2 --split test --model_dir .model/queryner/gliner2/best
```

---

## 项目结构

```
srs-ner-nlp/
├── main.py                  # CLI 入口（list / stats / show / train / validate）
├── .env.example             # WandB API key 模板（复制为 .env 并填入真实 key）
│
├── ner_datasets/            # NER 数据集封装模块
│   ├── __init__.py          # 公共导出
│   ├── base.py              # NERDataset 抽象基类、NERExample、DatasetStats
│   ├── queryner.py          # QueryNER 具体实现
│   └── registry.py          # DatasetRegistry 全局注册表
│
├── metrics/                 # 评估指标 & WandB 日志
│   ├── __init__.py          # 导出 NERMetrics, WandbConfig, WandbLogger
│   ├── base.py              # NERMetrics 数据类
│   └── wandb.py             # WandbLogger / WandbConfig
│
└── ner_trainer/             # NER 训练框架
    ├── __init__.py          # 导出所有公共接口
    ├── config.py            # BaseTrainConfig（框架无关通用配置）
    ├── base.py              # NERTrainer 抽象基类（通用训练骨架）
    ├── hanlp/               # HanLP 后端实现
    │   ├── __init__.py
    │   ├── config.py        # HanLPTrainConfig(BaseTrainConfig)
    │   └── trainer.py       # HanLPTrainer(NERTrainer)
    └── gliner2/             # GLiNER2 后端实现
        ├── __init__.py
        ├── config.py        # GLiNER2TrainConfig(BaseTrainConfig)
        └── trainer.py       # GLiNER2Trainer(NERTrainer)
```

临时生成目录（已在 `.gitignore`，不提交 git）：

```
.data/    # 训练数据 TSV 导出（由 NERDataset.export_tsv() 生成）
.model/   # 模型 checkpoint（按 <dataset_name>/epoch_NNN/ 和 best/ 组织）
```

---

## 核心抽象

### ner_datasets

**`NERExample`** — 单条样本，提供 `.entities()` 方法从 BIO 标签中提取实体 span：

```python
from ner_datasets.base import NERExample

ex = NERExample(
    tokens=["cascade", "platinum", "dishwasher", "pods"],
    labels=["B-creator", "B-product_name", "B-core_product_type", "I-core_product_type"],
)
print(ex.query)      # "cascade platinum dishwasher pods"
print(ex.entities()) # [('cascade', 'creator'), ('platinum', 'product_name'), ('dishwasher pods', 'core_product_type')]
```

**`NERDataset`** — 抽象基类，子类需实现 `load()` / `splits()` / `label_names()` / `iter_split()` 等接口。
新增 `export_tsv()` 方法，将任意 split 导出为 HanLP 兼容的两列 BIO TSV。

**`DatasetRegistry`** — 全局注册表，按名称管理所有数据集：

```python
from ner_datasets import registry

registry.list()           # 列出所有已注册数据集
ds = registry.get("queryner")
ds.load()
stats = ds.stats()        # 返回 DatasetStats（规模、标签分布等）
examples = ds.sample(split="test", n=5)
exported = ds.export_tsv(".data", splits=["train", "validation"])  # 导出 TSV
```

### metrics

**`NERMetrics`** — 统一评估结果数据类（`metrics/base.py`）：

| 字段 | 含义 |
|------|------|
| `precision` | 实体精确率 = TP / (TP + FP) |
| `recall` | 实体召回率 = TP / (TP + FN) |
| `f1` | 实体 F1，主排名指标 |
| `nb_correct / nb_pred / nb_true` | TP / (TP+FP) / (TP+FN) 原始计数 |
| `fp / fn` | 派生属性 |
| `case_accuracy` | 查询级别完全正确率（整条查询所有标签全对才算） |
| `nb_cases_correct / nb_cases_total` | case accuracy 分子/分母 |
| `loss` | 平均 cross-entropy loss（None 表示不可用） |
| `epoch` | 对应训练 epoch，独立 validate 时为 None |

所有指标均为 entity-level exact match（span 类型+边界全匹配），micro-average 聚合。

**`WandbLogger`** — WandB 日志记录器（`metrics/wandb.py`），密钥管理三层优先级：
1. 环境变量 `WANDB_API_KEY`（CI/CD）
2. 项目根目录 `.env` 文件（本地开发，不提交 git）
3. `wandb login` 写入的 `~/.netrc`

### ner_trainer

**`NERTrainer`** — 抽象基类（`ner_trainer/base.py`），封装通用训练骨架：
- epoch 循环
- 每 epoch 在 dev 集评估
- best checkpoint 管理（按指定指标选优，`copytree` 到 `best/`）
- 早停（`early_stopping_patience` 轮无改善则停止）
- 训练结束后加载 best checkpoint 在 test 集评估

子类只需实现四个钩子方法：

| 方法 | 职责 |
|------|------|
| `load_model()` | 初始化底层模型，赋值 `self.model` |
| `train_one_epoch(trn, dev, ckpt_dir, epoch)` | 跑一轮训练，保存 checkpoint |
| `evaluate(data_path, split, epoch)` | 评估，返回 `NERMetrics` |
| `load_from_checkpoint(ckpt_dir)` | 从目录恢复模型权重 |

**`BaseTrainConfig`** — 通用配置基类（`ner_trainer/config.py`）：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `dataset_name` | `"queryner"` | 数据集名称 |
| `train_split` | `"train"` | 训练 split |
| `dev_split` | `"validation"` | 验证 split |
| `test_split` | `"test"` | 最终评估 split |
| `data_dir` | `".data"` | TSV 导出目录 |
| `save_dir` | `".model"` | 模型保存目录 |
| `epochs` | `30` | 总训练轮数 |
| `best_metric` | `"f1"` | 选优指标（f1 / precision / recall / case_accuracy） |
| `early_stopping_patience` | `5` | 早停耐心值，≤0 表示禁用 |

**`HanLPTrainConfig`** — HanLP MTL 专属配置（`ner_trainer/hanlp/config.py`），在 BaseTrainConfig 基础上新增：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `transformer` | `"bert-base-cased"` | HuggingFace transformer 名称，作为 ContextualWordEmbedding 底座 |
| `average_subwords` | `True` | 是否对子词做平均池化（英文 WordPiece 场景建议开启） |
| `word_dropout` | `0.1` | Embedding dropout 概率 |
| `max_sequence_length` | `512` | Transformer 最大输入序列长度 |
| `batch_size` | `32` | 每个 batch 的样本数 |
| `lr` | `1e-3` | 任务头（decoder）学习率 |
| `encoder_lr` | `5e-5` | Transformer 编码器学习率（通常远小于任务头） |
| `grad_norm` | `5.0` | 梯度裁剪 max norm |
| `gradient_accumulation` | `1` | 梯度累积步数 |
| `warmup_steps` | `0.1` | Warmup 比例（<1.0）或绝对步数（≥1） |
| `tagging_scheme` | `None` | BIO 标注方案，None 自动推断 |
| `crf` | `False` | 是否使用 CRF 解码层 |
| `eval_trn` | `False` | 每 epoch 是否同时在训练集上评估 |

**`HanLPTrainer`** — HanLP MTL 后端实现（`ner_trainer/hanlp/trainer.py`）：

训练流程基于 `MultiTaskLearning` + `TaggingNamedEntityRecognition` + `ContextualWordEmbedding`，
参考官方 demo：[open_base.py](https://github.com/hankcs/HanLP/blob/master/plugins/hanlp_demo/hanlp_demo/zh/train/open_base.py)

```python
from ner_trainer.hanlp import HanLPTrainer, HanLPTrainConfig

config = HanLPTrainConfig(
    dataset_name="queryner",
    transformer="bert-base-cased",
    epochs=30,
    lr=1e-3,
    encoder_lr=5e-5,
    best_metric="f1",
    early_stopping_patience=5,
)
trainer = HanLPTrainer(config)
best_dir, dev_history, test_metrics = trainer.train()
print(test_metrics)

# 独立评估
metrics = trainer.validate(split="test")
```

Checkpoint 目录布局：

```
.model/queryner/
├── epoch_001/    ← 每 epoch 快照（HanLP fit 保存）
├── epoch_002/
├── ...
└── best/         ← 最优 checkpoint（validate 默认从此加载）
```

---

## 新增数据集

1. 在 `ner_datasets/` 下新建 `yourdata.py`，继承 `NERDataset` 并实现所有抽象方法：

```python
from .base import NERDataset, NERExample

class YourDataset(NERDataset):
    name = "yourdata"
    description = "..."
    hf_repo = "org/yourdata"

    def load(self): ...
    def splits(self): ...
    def label_names(self): ...
    def iter_split(self, split): ...
    def __len__(self): ...
    def split_len(self, split): ...
```

2. 在 `ner_datasets/registry.py` 末尾注册：

```python
from .yourdata import YourDataset
registry.register(YourDataset)
```

注册后 `uv run main.py list` 即可看到新数据集，所有子命令自动支持。

---

## 新增训练后端

1. 在 `ner_trainer/<backend>/` 下新建 `config.py` 和 `trainer.py`：

```python
# config.py
from ner_trainer.config import BaseTrainConfig

@dataclass
class YourConfig(BaseTrainConfig):
    your_field: str = "default"

# trainer.py
from ner_trainer.base import NERTrainer

class YourTrainer(NERTrainer):
    def load_model(self): ...
    def train_one_epoch(self, trn, dev, ckpt_dir, epoch): ...
    def evaluate(self, data_path, split, epoch=None) -> NERMetrics: ...
    def load_from_checkpoint(self, ckpt_dir): ...
```

epoch 循环、早停、WandB 上报、test 评估均由基类 `NERTrainer.train()` 自动处理。

---

## HanLP 预训练模型一览

> 官方文档：[hanlp.hankcs.com/docs](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/index.html)  
> HanLP 模型分为**多任务（mtl）**和**单任务**两大类。多任务模型速度快、显存省；单任务模型精度更高。

---

### 多任务模型（`hanlp.pretrained.mtl`）

多任务模型一次加载即可同时执行分词、词性、NER、句法分析等多个任务。

#### 中文模型（闭源语料，精度最高）

| 模型常量 | 编码器 | 说明 |
|----------|--------|------|
| `CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH` | ELECTRA-small | 闭源中文语料，任务：分词/词性/NER/SRL/依存/语义依存/成分句法（SD标准） |
| `CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH` | ELECTRA-base | 同上，base版精度更高 |
| `CLOSE_TOK_POS_NER_SRL_UDEP_SDP_CON_ELECTRA_SMALL_ZH` | ELECTRA-small | 闭源中文，依存使用UD标准；NER(MSRA) F1=96.05%，分词 F1=97.38% |
| `CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ERNIE_GRAM_ZH` | ERNIE-Gram-base | 闭源中文，百度ERNIE语义增强，词性/NER精度略优于ELECTRA |

#### 中文模型（开源语料，可用于商业）

| 模型常量 | 编码器 | 说明 |
|----------|--------|------|
| `OPEN_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH` | ELECTRA-small | 开源中文语料训练，授权更宽松 |
| `OPEN_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH` | ELECTRA-base | 同上，精度更高 |

#### 英文模型

| 模型常量 | 编码器 | 说明 |
|----------|--------|------|
| `EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_BASE` | ModernBERT-base | 英文联合模型，NER F1=84.67%，需 `transformers>=4.48` |
| `EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE` | ModernBERT-large | 同上large版，NER F1=87.11%，精度更高 |

#### 多语种模型（130种语言）

| 模型常量 | 编码器 | 支持语言 | 说明 |
|----------|--------|----------|------|
| `UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_MMINILMV2L6` | mMiniLMv2-L6 | 130种 | 轻量多语种，NER F1≈76.93%，适合资源受限场景 |
| `UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_MMINILMV2L12` | mMiniLMv2-L12 | 130种 | 多语种标准版，NER F1≈77.80% |
| `UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE` | XLM-R-base | 130种 | 多语种高精度版，NER F1≈80.34%，推荐跨语言场景 |

支持的主要语言包括：中文、英文、日文、法文、德文、俄文、阿拉伯文、印地文、西班牙文、葡萄牙文等130种，
完整列表见[官方文档](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/mtl.html#hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_MMINILMV2L6)。

#### 日文模型

| 模型常量 | 编码器 | 说明 |
|----------|--------|------|
| `NPCMJ_UD_KYOTO_TOK_POS_CON_BERT_BASE_CHAR_JA` | BERT-base-char-ja | 日文，训练于NPCMJ/UD/Kyoto语料，支持分词/词性/NER/依存/成分句法/SRL |

#### 古汉语模型

| 模型常量 | 编码器 | 说明 |
|----------|--------|------|
| `KYOTO_EVAHAN_TOK_LEM_POS_UDEP_LZH` | bert-ancient-chinese | 古汉语（文言文）分词/词元/词性/依存，分词 F1=99.01% |

---

### 单任务 NER 模型（`hanlp.pretrained.ner`）

单任务模型适合只需 NER 的场景。**以下为 HanLP 2.1.3 实际可用模型**（通过 `dir(hanlp.pretrained.ner)` 验证）：

| 模型常量 | 编码器 | 语言 | 语料 | 说明 |
|----------|--------|------|------|------|
| `MSRA_NER_ELECTRA_SMALL_ZH` | ELECTRA-small | 中文 | MSRA | F1=95.16，速度快，**项目默认使用** |
| `MSRA_NER_BERT_BASE_ZH` | BERT-base | 中文 | MSRA | 经典BERT，精度略高于ELECTRA-small |
| `MSRA_NER_ALBERT_BASE_ZH` | ALBERT-base | 中文 | MSRA | 轻量版BERT，参数少速度快 |

> **注意**：`CONLL03_NER_BERT_BASE_CASED_EN` 虽在 `hanlp.pretrained.ner` 中注册，但与 HanLP 2.1.3 不兼容（加载时报 `KeyError: 'average_subwords'`），**不可使用**。英文/多语种 NER 请使用多任务模型（mtl）。

---

### 本项目选型说明

`HanLPTrainConfig` 默认使用 `bert-base-cased` 作为 transformer 底座，当前 `scripts/hanlp_train_queryner.sh` 也沿用此模型。

- ELECTRA-small 速度快，适合训练迭代
- QueryNER 为**英文**电商查询，ELECTRA-small 基于中文预训练；若需英文友好底座，应改用多任务模型（mtl）后端，当前 HanLP 单任务 NER 无可用英文模型

| 场景 | 推荐模型 | 类型 |
|------|---------|------|
| 中文 NER，快速 | `MSRA_NER_ELECTRA_SMALL_ZH` | stl |
| 中文 NER，高精度 | `MSRA_NER_BERT_BASE_ZH` | stl |
| 英文 NER | `EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_BASE` | mtl（需改后端） |
| 多语种 NER | `UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE` | mtl（需改后端） |
| 古汉语 | `KYOTO_EVAHAN_TOK_LEM_POS_UDEP_LZH` | mtl |
| 日文 | `NPCMJ_UD_KYOTO_TOK_POS_CON_BERT_BASE_CHAR_JA` | mtl |
