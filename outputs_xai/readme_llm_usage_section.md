# LLM Usage Disclosure

In the interest of full transparency and research integrity, this section details the collaboration between the human researcher and LLM agents during the development of this project.

## 1. Scope of LLM Assistance

Large Language Models (LLMs) were utilized as interactive coding partners to accelerate development in the following areas:
*   **Planning and Architecture Design**: Structuring the seven phases of model experimentation.
*   **Code Review and Refactoring**: Identifying redundant operations in feature extraction and suggesting PEP 8 styling improvements.
*   **Leakage-Check Planning**: Designing the boundaries between train and validation sets, and outlining the forbidden features audit checks.
*   **Documentation Drafting**: Generating drafts for README sections and formatting tables.

## 2. Human Execution and Quality Control

All final modeling decisions and pipeline verification remain the sole responsibility of the human researcher:
*   **Local Code Execution**: All model training, feature extraction, evaluation metrics, and prediction outputs were generated on a local machine. No external LLM environment was used to run scripts or evaluate models.
*   **No Validation Label Exploitation**: No validation labels were shared with the LLM for hyperparameter optimization or tuning. All model selection decisions are based on local chronological validation output.
*   **Final Decision Authority**: The choice of linear models (Ridge and Logistic Regression) over complex ensembles (Random Forest and Gradient Boosting) was a human decision motivated by safety, generalization, and auditability.
*   **Reproducibility**: The final code (`solution.py`) is fully self-contained and reproducible, running from scratch without any dependency on LLM APIs or interactive prompts.
