<div align="center">

# KoNET: Korean National Education Test
[![Paper](https://img.shields.io/badge/Paper-arxiv.2502.15422-orange)](https://arxiv.org/abs/2502.15422)
[![Conference](https://img.shields.io/badge/NAACL-2025-red)]()

Official Implementation of KoNET | [Paper](https://arxiv.org/abs/2502.15422)
</div>

## Introduction
**KoNET**, **Ko**rean **N**ational **E**ducation **T**est, a new benchmark designed to evaluate Multimodal Generative AI Systems using Korean national educational tests. KoNET consists of four levels of exams, corresponding to elementary, middle, high school, and college levels. These exams are renowned for their rigorous standards and diverse questions, facilitating a comprehensive analysis of AI performance across different educational levels. 

Our academic paper, which describes our method in detail and provides full experimental results and analyses, can be found here:<br>
> [**Evaluating Multimodal Generative AI with Korean Educational Standards**](https://arxiv.org/abs/2502.15422).<br>
> [Sanghee Park](https://scholar.google.com/citations?user=_ryVHp0AAAAJ) and [Geewook Kim](https://geewook.kim). To appear at NAACL 2025.

## Software Installation
- Requires-Python >=3.9

```bash
git clone https://github.com/naver-ai/konet.git
cd KoNET
pip install -r requirements.txt
```

## KoNET Generation Guide
```python
from src.generator import generate_KoNET
dataset = generate_KoNET()
display(dataset)
```
```bash
[1/3] Downloading files: 100%|██████████| 72/72 [00:17<00:00,  4.19it/s]
[2/3] Converting PDFs to images: 100%|██████████| 100/100 [01:41<00:00,  1.02s/it]
[3/3] Extracting problem images: 100%|██████████| 2377/2377 [02:28<00:00, 15.97it/s]
```

## KoNET Evaluation Guide
- Submission File:
  - Please complete the  [submission file](src/utils/submission.json).
  - Please refer to the [submission_test file](src/utils/submission_test.json) for guidance.
- Cost Estimation (excluding vision tokens):
    - Based on `gpt-4o-mini-2024-07-18` (default):
        - Average Input Length: 2,000 characters
        - Average Output Length: 10 characters
        - Price per API Call: $0.0001
        - Estimated Cost for 2,377 Responses: ~$0.18
    - Comparison with Other Models:
        - `gpt-4o-2024-08-06`: $0.0013 per call → ~$3.04
        - `o1-2024-12-17`: $0.0077 per call → ~$18.26
    - The actual cost may vary from this estimate.
- Result Variability:
    - Even with the same responses, results may vary depending on judgement API parameters (e.g., temperature, top_p, seed, etc.).
- This benchmark includes questions from the "Listening" section, all of which will be marked as correct.

```python
import os
os.environ["OPENAI_KEY"] = "YOUR OPENAI KEY"
```
```python
from src.evaluator import evaluate_KoNET
result = evaluate_KoNET(submission_filepath="src/utils/submission_test.json")
print(result)
```
```bash
Evaluating submissions: 100%|██████████| 2377/2377 [02:38<00:00, 14.99it/s]
==============================
KoNET Acc: 51.89% (55/106)
- KoEGED Acc: 75.00% (15/20)
- KoMGED Acc: 65.00% (13/20)
- KoHGED Acc: 60.00% (12/20)
- KoCSAT Acc: 32.61% (15/46)
==============================
```

## How to Cite
If you find our work useful in your work, please consider citing our paper:
```
@misc{park2025evaluatingmultimodalgenerativeai,
      title={Evaluating Multimodal Generative AI with Korean Educational Standards}, 
      author={Sanghee Park and Geewook Kim},
      year={2025},
      eprint={2502.15422},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.15422}, 
}
```

## License
```
KoNET
Copyright (c) 2025-present NAVER Cloud Corp.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
```
