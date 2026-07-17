# Dataset Download Guide

This directory contains datasets required for ReMe evaluation. Some datasets are large and excluded from Git version control — they must be downloaded manually.

## LongMemEval

The LongMemEval dataset is hosted on HuggingFace and can be downloaded via the provided script (uses hf-mirror.com).

```bash
cd datasets/longmemeval

# Download all data files (existing files will be skipped automatically)
python download.py

# Download only longmemeval_m_cleaned.json
python download.py --m-only
```

After downloading, the directory should contain:

| File | Description |
| --- | --- |
| `longmemeval_oracle.json` | Oracle dataset |
| `longmemeval_s_cleaned.json` | S-scale dataset |
| `longmemeval_m_cleaned.json` | M-scale dataset |
| `final_groundtruth_cleaned_s.json` | Cleaned ground truth for S-scale (included in repo) |
| `download.py` | Download script (included in repo) |

> **Note**: The download script uses hf-mirror.com by default. To use a different mirror, modify `BASE_URL` in `download.py`.

## BEAM

BEAM is a public repository. Clone it directly into the `datasets/` directory:

```bash
cd datasets
git clone https://github.com/mohammadtavakoli78/BEAM.git
```

After cloning, `datasets/BEAM/` should contain `chats/`, `src/`, `topics/` and other subdirectories.
