This repository contains the code and datasets accompanying the research paper: **"Need-Oriented Response Generation for Navigating the Bias–Utility Trade-off in LLMs"**

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
```


