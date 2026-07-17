# 数据集下载说明

本目录包含 ReMe 评测所需的数据集。部分数据集体积较大，不纳入 Git 版本管理，需要手动下载。

## LongMemEval

LongMemEval 数据集需要从 HuggingFace 镜像下载。

```bash
cd datasets/longmemeval

# 下载全部数据文件（已存在的文件会自动跳过）
python download.py

# 仅下载 longmemeval_m_cleaned.json
python download.py --m-only
```

下载完成后，目录下应包含以下文件：

| 文件名 | 说明 |
| --- | --- |
| `longmemeval_oracle.json` | Oracle 数据集 |
| `longmemeval_s_cleaned.json` | S 规模数据集 |
| `longmemeval_m_cleaned.json` | M 规模数据集 |
| `final_groundtruth_cleaned_s.json` | S 规模清洗后的 ground truth（已随仓库提供） |
| `download.py` | 下载脚本（已随仓库提供） |

> **注意**：下载脚本使用 hf-mirror.com 镜像源，如需更换源请修改 `download.py` 中的 `BASE_URL`。

## BEAM

BEAM 数据集为公开仓库，直接 clone 到 `datasets/` 目录下即可：

```bash
cd datasets
git clone https://github.com/mohammadtavakoli78/BEAM.git
```

clone 完成后，`datasets/BEAM/` 目录下应包含 `chats/`、`src/`、`topics/` 等子目录。
