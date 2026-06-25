================================================================================
          PROJECT: AI-POWERED SEMANTIC RESUME-JOB MATCHING SYSTEM
================================================================================

ABSTRACT
This project addresses the inefficiencies of traditional Applicant Tracking 
Systems (ATS) that rely on keyword matching (TF-IDF), which often penalize 
qualified candidates for lexical variations. We developed a multi-stage deep 
learning pipeline that utilizes Sentence-BERT (SBERT) to generate high-dimensional 
contextual embeddings, capturing the "latent meaning" of professional skills. 
The system further implements a supervised neural refinement layer using a 
Siamese-style architecture, which achieved 96% accuracy and 1.00 ROC-AUC in 
discriminating between valid and invalid matches. Results demonstrate a 122% 
increase in semantic alignment compared to lexical baselines, providing a 
robust, scalable decision-support framework for intelligent recruitment.

TECHNICAL REQUIREMENTS
- Python 3.8+
- Libraries: torch, sentence-transformers, pandas, numpy, sklearn, matplotlib, seaborn
- Hardware: GPU recommended for SBERT encoding (Google Colab/Kaggle supported)

HOW TO RUN THE CODE
The project is provided as a Jupyter Notebook (.ipynb). Follow these steps:

1. Environment Setup:
   Install dependencies via pip:
   $ pip install torch sentence-transformers pandas scikit-learn seaborn wordcloud

2. Dataset Placement:
   Ensure the following directory structure exists in your working directory:
   /Resume/Resume.csv
   /JobDescription/job_title_des.csv

3. Execution:
   Open the notebook in Jupyter Lab/Notebook or Google Colab and "Run All".
   The script executes through five distinct phases:
   - Phase 1-2: Preprocessing & Cleaning (Stripping HTML, lowercasing)
   - Phase 3: Baseline Generation (TF-IDF Sparse Vectorization)
   - Phase 4: Semantic Embedding (SBERT all-MiniLM-L6-v2 Encoding)
   - Phase 5: Supervised Fine-Tuning (Neural MatchModel training)

4. Command Line Usage (If converted to .py):
   $ python matching_system.py --mode train --model all-MiniLM-L6-v2

   Note: The code defaults to IT category filtering to ensure domain-specific 
   accuracy. You can modify 'Category' filters in the data loading section to 
   test other job sectors (e.g., HR, Sales, Healthcare).

EXPERIMENTAL RESULTS SUMMARY
- TF-IDF Mean Similarity: 0.158
- SBERT Mean Similarity: 0.351 (Significant Rightward Shift)
- Neural Classifier F1-Score: 0.96
- Ranking Agreement (SBERT vs Neural): 56.7% (Neural model refined 43% of cases)

================================================================================
