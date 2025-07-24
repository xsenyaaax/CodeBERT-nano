---
title: Code Snippet Prediction
emoji: 🧠
colorFrom: indigo
colorTo: pink
sdk: docker
app_file: Dockerfile
pinned: false
short_description: Code Snippet Language Prediction using Transformers
---

# Transformer-Based Code Snippet Classifier

This project implements a model with a transformer architecture for classifying code snippets using a custom tokenizer and PyTorch. It includes training, evaluation, and visualization of attention heads for interpretability.

## Overview

* Custom tokenizer training using the Hugginface library
* Transformer model implemented from scratch using PyTorch and following Andrej Karpathy's [videos](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) <3
* Demo build with [Gradio](https://www.gradio.app/) library mounted on FastAPI app with attention visualization via [BertViz](https://github.com/jessevig/bertviz)
* What models are implemented is described in [**Models section**](#models) section.

## Example

Below is a screenshot of the demo app in action:
![app-example](static/app-example.png)


Attention visualization page (sorry for the page looking so bad, I am not that good with CSS :()
![attention example](static/attention-visualize.png)

## Dataset
The dataset used is publicly available [here](https://figshare.com/articles/dataset/Code_Snippets_Dataset/22623331?file=40146943). This dataset contains labeled code snippets used for classification tasks. It is also available in `/datasets` folder.

All available languages to predict from: 
* c
* c++
* css
* html
* java
* javascript
* python
* r
* sqlite

## Notes

* This project was done for experimentation and learning. It does not use any pretrained big models.
* The tokenizer retains readable words to make attention visualization more interpretable.
* Goal was not build super deep and big model but to experiment, therefore training was done on CPU 

## Models

### Model Version and Progress

Throughout the project, multiple versions of the model were developed, each adding complexity and performance improvements. Models share that they predict based on `[CLS]` token inserted during the tokenization on first position in the code snippet. 

Every model implementation you can find in the `notebooks/transformer.ipynb` jupyter notebook with analysis of the training in the end. 

###  Model Comparison Table

| Version                   | Description                                             | Accuracy on Validation Data| Complexity   | Throughput |
|---------------------------|---------------------------------------------------------|--------------|--------------|------------|
| Basic Model V1               | Embedding + linear classifier                          | ~11.0%        | Low          | ~105k tokens/sec       |
| Position Embedding Model V2 | Embedding + positional encoding + linear               | ~11.0%        | Low-Medium   | ~100k tokens/sec       |
| Attention-Based Model     | Embedding + positional encoding + 1 Self-attention head + 1 linear        | 92%        | Medium       | ~65k tokens/sec     |
| Multi-Head Attention-Based Model     | Embedding + positional encoding + Multiple Self-attention heads + 1 linear        | 91%        | Medium-High      | ~40k tokens/sec     |
| Full Transformer Encoder  | Multi-layer encoder with attention, FFN, and residuals | 87.6%        | High         | ~20k tokens/sec        |