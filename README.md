# Overview

In large open-source ecosystems (e.g., Eclipse, Mozilla, Apache), development teams receive hundreds of unstructured defect submissions daily. Manual bug triage—assigning priority levels (P1 Blocker to P5 Trivial) and identifying duplicate reports—is time-consuming, subjective, and prone to error.
This project delivers an **end-to-end, trustworthy, multi-task AI framework** for automated bug triage:
1. **Multi-Class Priority Classification (P1–P5):** Unsupervised Topic Modeling (LDA, $k=10$) coupled with specialized Topic-BERT Transformers trained under cost-sensitive class weighting to overcome severe majority-class imbalance (87.9% P3 vs ~8.4% P1/P2).
2. **Sub-10ms Semantic Duplicate Retrieval:** Siamese Sentence-BERT (SBERT) paired with Meta FAISS dense vector indexing to detect duplicate defects submitted with differing vocabulary.
3. **Model Explainability (XAI):** SHAP and LIME token attribution heatmaps explaining the root-cause justification behind every AI priority recommendation.


# 1. Clone repository
git clone https://github.com/8rachana/Explainable-Multi-Task-Bug-Triage-System.git
cd Explainable-Multi-Task-Bug-Triage-System

# 2. Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
