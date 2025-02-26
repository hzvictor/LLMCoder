# LLMCoder: Enhancing Medical Coding Efficiency through Domain-Specific Fine-Tuned Large Language Models

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)

## Overview

LLMCoder is a novel framework that employs domain-specific fine-tuned Large Language Models (LLMs) to automate ICD-10 medical coding. Our approach demonstrates that LLMs, when properly fine-tuned with comprehensive medical coding knowledge, can achieve high accuracy in translating clinical documentation into standardized ICD-10 codes.

## Key Features

- **Two-Stage Fine-Tuning Pipeline**: Initial fine-tuning with complete ICD-10 code set (74,260 code-description pairs) followed by enhanced fine-tuning addressing linguistic and lexical variations
- **High Accuracy**: Achieves 71.64% exact code match and 88.85% category match on real-world clinical notes
- **Adaptability**: Effectively handles common clinical documentation challenges:
  - Reordered diagnostic expressions (87.51%)
  - Medical abbreviations (96.59%)
  - Typographical errors (94.18%)
  - Multiple concurrent conditions (98.04%)
- **Model Options**: Supports both proprietary (GPT-4o mini) and open-source (Llama) models
- **Local Deployment**: Optimized for resource-constrained environments (Llama-3.2-1B requires only 2GB GPU memory)

## Results

| Model Type | Base Scenario | Reordered Expressions | Abbreviations | Typos | Multiple Conditions | Clinical Notes |
|------------|---------------|----------------------|---------------|-------|---------------------|----------------|
| Pre-trained | <1% | <4% | <4% | <4% | <4% | <3% |
| Initial Fine-tuned | >97% | 36-78% | >92% | 58-84% | 3-11% | <0.1% |
| Enhanced Fine-tuned | >97% | >85% | >95% | >93% | >94% | 71.64% |

## Repository Structure

This repository contains the following key Jupyter notebooks that demonstrate our workflow:

### Notebooks

| Notebook | Description |
|----------|-------------|
| `1_data_preparation.ipynb` | Generates training and testing datasets for all scenarios, including linguistic and lexical variations |
| `2_initial_finetuning.ipynb` | Implements the initial fine-tuning stage using the complete ICD-10 code set |
| `3_enhanced_finetuning.ipynb` | Implements the enhanced fine-tuning stage for handling linguistic and lexical variations |
| `4_clinical_notes_evaluation.ipynb` | Evaluates model performance on real-world clinical notes from MIMIC-IV |
| `5_error_analysis.ipynb` | Analyzes error patterns and generates visualizations of model performance |

### Example Data

The repository includes sample data for demonstration:
- `data/icd10_sample.csv`: Sample of the ICD-10 code set
- `data/variations/`: Examples of each variation type
- `data/clinical_notes_samples/`: Anonymized sample clinical notes (without PHI)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/hzvictor/LLMCoder.git
cd LLMCoder

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebooks

1. Start Jupyter Lab:
```bash
jupyter lab
```

2. Open and run the notebooks in sequence (1-5) to replicate our workflow.

3. For model inference on new data:
```python
# Load the model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("path/to/saved/model")
model = AutoModelForCausalLM.from_pretrained("path/to/saved/model")

# Prepare prompt
prompt = f"""You are a medical coding specialist responsible for assigning ICD-10 codes to clinical documentation.

Generate appropriate ICD-10 codes from the following clinical note:

{your_clinical_note}"""

# Generate prediction
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=512)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## Datasets

### Data Sources
This project uses the following data sources:

1. **ICD-10-CM Code Set**: Complete set of 74,260 code-description pairs used for initial fine-tuning
2. **Synthetic Variations**: Generated data for each linguistic and lexical variation type:
   - Reordered diagnostic expressions
   - Typographical errors
   - Medical abbreviations
   - Multiple concurrent conditions
   - Paraphrased diagnostic contexts
3. **MIMIC-IV Dataset**: 20,000 discharge summaries used for real-world evaluation

### Data Generation Process
Our data generation process is documented in `1_data_preparation.ipynb`, which:
- Creates high-quality example pairs for each variation type
- Generates training and testing datasets via API calls
- Ensures clinical authenticity and coding complexity

*Note: Due to data privacy restrictions, the MIMIC-IV dataset is not included in this repository. Researchers must [apply for access](https://physionet.org/content/mimiciv/2.2/) separately.*

## Model Training Details

- **Hardware**: 4x NVIDIA A100 GPUs (80GB each)
- **Software**: Llama Factory with DeepSpeed optimization
- **Training Parameters**:
  - Learning rate: 1e-5
  - Optimizer: AdamW with cosine scheduler
  - Sequence length: 50 tokens
  - Epochs: 10

## Citation

If you use LLMCoder in your research, please cite our paper:

```bibtex
@article{hou2025enhancing,
  title={Enhancing Medical Coding Efficiency through Domain-Specific Fine-Tuned Large Language Models},
  author={Hou, Zhen and Liu, Hao and Bian, Jiang and He, Xing and Zhuang, Yan},
  journal={},
  year={2025},
  publisher={}
}
```

## Team

- Zhen Hou, M.S. - Indiana University, Indianapolis
- Hao Liu, Ph.D. - Montclair State University
- Jiang Bian, Ph.D. - Indiana University, Indianapolis
- Xing He, Ph.D. - Indiana University, Indianapolis
- Yan Zhuang, Ph.D. - Indiana University, Indianapolis

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We gratefully acknowledge the support of Indiana University and the use of the MIMIC-IV database. This research was conducted using computing resources provided by the Department of Biomedical Engineering and Informatics, Luddy School of Informatics, Computing, and Engineering.
