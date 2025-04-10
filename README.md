[![CC BY license](https://img.shields.io/badge/License-CC%20BY-lightgray.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/python-3.11-gold.svg)](https://www.python.org/downloads/release/python-311/)
[![PyTorch](https://img.shields.io/badge/Pytorch-2.0-pumpkin.svg)](https://pytorch.org/get-started/previous-versions/#v200)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models-purple)](https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending)

# 👀 What?
This repository contains code for implementing models and evaluating them for the Visual Storytelling task (using the VWP[^1] dataset):  
**[On the Challenges in Evaluating Visually Grounded Stories]()**&mdash;In proceedings of the [Text2Story](https://text2story25.inesctec.pt/) workshop (ECIR 2025).

**Note:** Despite being proposed specifically for visual storytelling, this method is generalizable and can be extended to any task involving model-generated outputs with corresponding references.

# 🤔 Why?
VWP dataset is constructed using scenes from movies. Compared to the popular VIST dataset:
- Visual sequences in VWP are well-connected and centered around recurring characters
- Stories are longer with diverse entities

The recently proposed $d_{HM}$[^2] metric evaluates model-generated stories by measuring their closeness to human stories along three dimensions&mdash;*Coherence, Visual grounding, Repetition*

In this work, we use the $d_{HM}$ metric and compare several general-purpose foundation vision-language-models (VLMs) with models trained specifically on the VWP`v2.1` dataset. We discuss their performance, underline the challenges in evaluating the visually-grounded stories, and argue for considering more dimensions important for automatic narrative generation.

# 🤖 How?
For generating stories using VLMs, use the following code:  
`pip install -r requirements.txt`  
`python -u generate_stories.py --model qwen-vl` (run `python generate_stories.py --help` for more options)

For training & generating stories using the `TAPM (+LLAMA 2)` model and for evaluating stories using the $d_{HM}$ metric, we followed the instructions in [this repository](https://github.com/akskuchi/dHM-visual-storytelling).


[^1]: https://aclanthology.org/2023.tacl-1.33
[^2]: https://aclanthology.org/2024.findings-emnlp.679

---
🔗 If you find this work useful, please consider citing it:
```
@inproceedings{
}
```
