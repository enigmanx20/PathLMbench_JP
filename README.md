# PathLMbench_JP

PathLMbench_JP is a small toolkit and a set of templates for benchmarking processing and typo-correction workflows on Japanese pathology reports. It provides Jupyter notebooks for dataset conversion and benchmarking, utilities for typo injection and evaluation, and example server commands for running local LLM models.

**Paper**

- Japanese original paper: under submission  
- English preprint: [arXiv:2603.11597](https://arxiv.org/abs/2603.11597)


**Notebooks**
- [A1_template_breast.ipynb](A1_template_breast.ipynb): Convert JSON dataset into the project's local format.
- [A2_template_breast.ipynb](A2_template_breast.ipynb): Conversion with added pT calculation and scoring.
- A3 Converts local format into Kiyaku format with an example. The notebook is not distributed because of copyright concern. 
- [A4_template_breast.ipynb](A4_template_breast.ipynb): Extract local-format records and re-export to JSON.
- [B_typo.ipynb](B_typo.ipynb): Notebook for typo insertion experiments and evaluation.
- [C_explanation.ipynb](C_explanation.ipynb): Human-evaluation and explanation workflows.

**Key scripts and helpers**
- [typo_utils.py](typo_utils.py): Core utilities for tokenization (MeCab), normalization, n-gram evaluation, and typo-generation/evaluation. This file contains the primary implementations used by the notebooks.

- [llama-server_command.md](llama-server_command.md): Example `llama-server` commands for running local models used in experiments.

**Data & dictionaries**
- `dictionaries/` contains example resources used for MeCab:
	- [dictionaries/user.dic](dictionaries/user.dic): used in combination with a general dictionary (e.g. mecab-ipadic-2.7.0-20070610). 
	- [dictionaries/word_dict.csv](dictionaries/word_dict.csv): original CSV for user.dic

- [typo_utils.py](typo_utils.py): helper utilities for typo insertion and evaluation.
- Template notebooks: several Jupyter notebooks for running and recording benchmarks.

**Requirements**
- Install system dependency MeCab and an IPADIC-compatible dictionary (e.g. mecab-ipadic-2.7.0-20070610).
- Python packages (example):

```bash
python3 -m pip install pandas tqdm jaconv mecab-python3
```

**Usage**
1. Clone the repository

```bash
git https://github.com/enigmanx20/PathLMbench_JP.git
cd PathLMbench_JP
```

2. Install system and python dependencies including llama.cpp (https://github.com/ggml-org/llama.cpp)

3. Running a local LLM server for generation/benchmarking

Refer to `llama-server_command.md` for several example `llama-server` commands. A minimal example (adjust `-m`/`--model` to your model path):

4. Input files for notebook workflows
Locate mecab dictionaries in the dictionaries folder.
- `reports.tsv`: if you want to benchmark on local pathology reports, prepare a TSV/CSV and adjust column names in the notebooks (there are cells that load and rename columns — search for `reports.tsv`in the notebooks).

5. Run the notebooks

- Start Jupyter Lab and open the notebooks listed above (run cells in order where applicable):

```bash
jupyter lab
```
## Table1: Prompt processing and text generation llama-bench results

| Model | Quantization | Thinking | Additional MK | Size (GB) | Params (B) | t/s (pp512) | t/s (tg128) |
|------|--------------|----------|----------------|-----------|-------------|--------------|--------------|
| Gemma 3-27b-it | Q4_0 |  |  | 16.04 | 27.01 | 384.66 ± 0.17 | 29.90 ± 0.60 |
| MedGemma-27b-text-it | Q4_K_XL |  | ✅ | 15.66 | 27.01 | 337.32 ± 0.16 | 27.31 ± 0.83 |
| SIP-jmed-llm-3-8x13b-AC-32k-instruct | Q8_0 |  | ✅ | 72.40 | 73.16 | 421.82 ± 0.64 | 25.97 ± 0.61 |
| Qwen3-Next-80B-A3B-Instruct | Q8_0 |  |  | 78.98 | 79.67 | 647.65 ± 5.06 | 24.97 ± 0.03 |
| Qwen3-Next-80B-A3B-Thinking | Q8_0 | ✅ |  | 78.98 | 79.67 | 650.06 ± 2.92 | 24.94 ± 0.04 |
| Gpt-Oss-20b | Native MXFP4 | ✅ |  | 11.27 | 20.91 | 2293.59 ± 7.69 | 120.27 ± 0.19 |
| Gpt-Oss-120b | Native MXFP4 | ✅ |  | 59.02 | 116.83 | 1170.35 ± 8.01 | 80.03 ± 0.17 |
| Qwen3.5-27B | A8_0 | ✅ |  | 26.62 | 26.90 | 389.03 ± 0.58 | 20.67 ± 0.01 |
| Qwen3.5-27B | UD-Q4_K_XL | ✅ |  | 15.57 | 26.90 | 335.35 ± 0.19 | 25.53 ± 0.03 |
| Gemma-4-31B-it | Q4_0 | ✅ |  | 16.13 | 30.70 | 332.55 ± 0.24 | 28.43 ± 0.62 |


