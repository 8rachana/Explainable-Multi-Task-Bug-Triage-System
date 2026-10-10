# Overview

In large open-source ecosystems (e.g., Eclipse, Mozilla, Apache), development teams receive hundreds of unstructured defect submissions daily. Manual bug triage—assigning priority levels (P1 Blocker to P5 Trivial) and identifying duplicate reports—is time-consuming, subjective, and prone to error.

**Phase 1 Objective:**  
Scientifically replicate and benchmark the state-of-the-art methodology published by **Riley & Moin (April 2025, arXiv:2504.15912)**

# 1. Clone repository
git clone https://github.com/8rachana/Explainable-Multi-Task-Bug-Triage-System.git
cd Explainable-Multi-Task-Bug-Triage-System

# 2. Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Benchmark Evaluation & Metrics
To evaluate the models and view the  performance metrics (Macro-F1, Precision, Recall, Accuracy, and Per-Class F1 comparison against the base paper across 17,032 test bug reports), run:

python show_benchmark_results.py
