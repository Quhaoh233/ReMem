# ReMem

Official implementation and evaluation code for **ReMem:Rethinking Perception and Memory in Long-Context Recommendation Agents**.

<img width="3696" height="1845" alt="5125aa6e-5d6a-4483-9614-53ebfe9d691b" src="https://github.com/user-attachments/assets/929c74cc-4d35-4c86-9eb0-5f5db67e0242" />


*A simple method is not necessarily the best, but a strong method should remain as simple as possible.*
ReMem follows this principle by realizing long-context preference modeling through sequential memory updates within the standard autoregressive process, without introducing external retrieval systems, specialized memory modules, or architectural modifications.
(More updates are coming soon, including model checkpoints, training code, and more.)

## Repository Structure

```text
.
├── code/                 # Source code and evaluation scripts
│   ├── main.py           # Main entry point
│   └── .env              # OpenAI API configuration (create locally)
├── data/                 # Dataset directory
├── requirements.txt      # Python dependencies
└── README.md
```

## Dataset

Download the datasets from [Google Drive](https://drive.google.com/drive/folders/1qkrbIY5zy0L0TRKNOlM4Pzk6XwuEeqtf?usp=sharing) and place the downloaded files in the `data/` directory:

```text
data/
└── [downloaded dataset files]
```

If the directory does not exist, create it from the repository root:

```bash
mkdir -p data
```

## Environment Setup

We recommend using **Python 3.12** with the **Unsloth** environment.

Alternatively, install all required dependencies using:

```bash
pip install -r requirements.txt
```

Using a virtual environment is recommended:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Configuration

### Hugging Face Token

Enter your [Hugging Face access token](https://huggingface.co/settings/tokens) in `code/main.py` to access open-source models such as:

- Qwen3.5-9B
- Qwen3-VL-8B

Ensure that your Hugging Face account has permission to access any gated models used in the experiments.

### OpenAI API Key

The LLM-as-a-Judge evaluation requires an OpenAI API key. Create `code/.env` and add your key as follows:

```env
OPENAI_API_KEY=your_openai_api_key
```

> **Security:** Do not commit API keys or access tokens to GitHub. Ensure that `.env` is included in `.gitignore`.

## Quick Start

Run the main experiment with:

```bash
cd code
python main.py
```

Additional configuration options are available through the argument parser defined in `main.py`. To inspect the supported arguments, run:

```bash
python main.py --help
```

## Citation

If you find this repository useful, please cite our work:

```bibtex
@article{qu2026remem,
  title={ReMem: Rethinking Perception and Memory in Long-Context Recommendation Agents}, 
  author={Haohao Qu and Yongcheng Jing and Chun Hin Chan and Shanru Lin and Wenqi Fan and Dacheng Tao},
  year={2026},
  journal = {arXiv preprint},
}
```
