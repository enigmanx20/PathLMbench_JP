# PathLMbench_JP

PathLMbench_JP is a small toolkit and a set of templates for benchmarking processing and typo-correction workflows on Japanese pathology reports. It provides Jupyter notebooks for dataset conversion and benchmarking, utilities for typo injection and evaluation, and example server commands for running local LLM models.

**Paper**

- Japanese original paper: under submission  
- English preprint: https://arxiv.org/abs/2603.11597
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

