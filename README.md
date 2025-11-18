# Need-Oriented AI: Nonviolent Communication (NVC) for Unbiased & Useful LLMs

This repository contains the code and datasets accompanying the research paper: **"Navigating the Bias-Utility Trade-off in Conversational AI with Nonviolent Communication."**

## Abstract

Efforts to create unbiased conversational AI often result in a significant loss of utility. Instead of providing actionable answers, these systems frequently resort to diplomatic but ambiguous rejections, leaving users caught between potentially biased systems and safe but unhelpful ones. 

This paper introduces a novel framework that navigates this trade-off by incorporating the principles of **Nonviolent Communication (NVC)** into Large Language Models. Instead of a single-step refusal, our approach operationalizes NVC’s four-component structure (**Observation, Feeling, Need, Request**) to reframe sensitive queries, focusing on the user’s underlying needs to "discuss and deliver" a constructive response. 

We conduct a comparative study of four prompt-induced behaviours across three local LLMs, evaluating performance using an LLM-as-a-judge framework. Results show our NVC-prompted model significantly reduces bias while simultaneously maintaining or improving utility and user satisfaction scores compared to standard and fairness-focused baselines. This work presents a practical, reproducible method for developing a more "need-oriented" conversational AI that preserves utility not by avoiding difficult topics, but by engaging with them constructively.

## Repository Structure

The repository is organized into three main directories:

```text
.
├── data/                     # Source data for bias testing
│   └── BiasAsker/
│     └── [bias_asker_combined.csv]
│   └── Response_generation_datasets/
│     ├── [Mental-Health-GPT-Augmented-Clean-Dataset.csv]
│     └── [Statement_Need_Dataset_GPT_100.csv]
├── source/ 
│   └── prompted_models/
│     ├── [NVC_Pipeline.ipynb]
│     ├── [unbiased_code.py]
│     ├── [utility_first_code.py]
│     └── [vanilla_code]
├── code/                         
│   ├── inference.py
│   ├── evaluation_judge.py
│   └── ...
└── README.md
1. Response generation datasets
This folder contains the datasets generated during our comparative study. It includes the raw outputs from the three local LLMs tested across four different prompting strategies (including our NVC method and standard baselines). These files are formatted for analysis by the LLM-as-a-judge framework.

2. BiasAsker
This folder contains the BiasAsker dataset used as the primary input for stress-testing the models for social bias. It includes the specific sensitive queries and triggers used to evaluate the safety and utility trade-offs of the models.

3. code
This directory contains the implementation of our framework:

Prompting Templates: The specific system prompts used to operationalize the NVC components (Observation, Feeling, Need, Request).

Inference Scripts: Code to run the local LLMs and generate responses.

Evaluation: The "LLM-as-a-judge" scripts used to score the responses on bias, utility, and user satisfaction.
