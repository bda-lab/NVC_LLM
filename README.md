This repository contains the code and datasets accompanying the research paper: **"Navigating the Bias-Utility Trade-off in Conversational AI with Nonviolent Communication."**

## Abstract

Efforts to create unbiased conversational AI often result in a significant loss of utility. Instead of providing actionable answers, these systems frequently resort to diplomatic but ambiguous rejections, leaving users caught between potentially biased systems and safe but unhelpful ones. 

This paper introduces a novel framework that navigates this trade-off by incorporating the principles of **Nonviolent Communication (NVC)** into Large Language Models. Instead of a single-step refusal, our approach operationalizes NVC’s four-component structure (**Observation, Feeling, Need, Request**) to reframe sensitive queries, focusing on the user’s underlying needs to "discuss and deliver" a constructive response. 

We conduct a comparative study of four prompt-induced behaviours across three local LLMs, evaluating performance using an LLM-as-a-judge framework. Results show our NVC-prompted model significantly reduces bias while simultaneously maintaining or improving utility and user satisfaction scores compared to standard and fairness-focused baselines. This work presents a practical, reproducible method for developing a more "need-oriented" conversational AI that preserves utility not by avoiding difficult topics, but by engaging with them constructively.

## Repository Structure

The repository is organized into three main directories:

```text
.
├── BiasAsker/                     # Source data for bias testing
│   └── [BiasAsker Dataset Files]
├── Response generation datasets/  # Generated responses from the 3 local LLMs
│   ├── [Baseline Responses]
│   └── [NVC-Prompted Responses]
├── code/                          # Scripts for inference, prompting, and evaluation
│   ├── inference.py
│   ├── evaluation_judge.py
│   └── ...
└── README.md
```

1. Response generation datasets
This folder contains the datasets used to generate responses for our comparative study. 

2. BiasAsker
This folder contains the BiasAsker dataset used as the primary input for stress-testing the models for social bias. It includes the specific sensitive queries and triggers used to evaluate the safety and utility trade-offs of the models.

3. code
This directory contains the implementation of our framework:

Prompting Templates: The specific system prompts used to operationalize the NVC components (Observation, Feeling, Need, Request).

Inference Scripts: Code to run the local LLMs and generate responses.

Evaluation: The "LLM-as-a-judge" scripts used to score the responses on bias, utility, and user satisfaction.

## Methodology: The NVC Framework
Our approach moves away from binary refusal mechanisms by applying Rosenbergs's Nonviolent Communication components to prompt engineering:

Observation: The model objectively states what is seen in the prompt without evaluation.

Feeling: The model identifies potential emotional contexts or sensitivities.

Need: The model identifies the user's underlying informational or functional need (the "why" behind the query).

Request/Response: The model delivers a constructive response that satisfies the Need while neutralizing the Bias.
