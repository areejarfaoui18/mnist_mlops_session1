# MNIST MLOps Mentorship Starter

This is a deliberately simple PyTorch image-classification project for the first mentorship session.

The goal is NOT to study computer vision. The goal is to use a small, runnable ML project and progressively turn it into a production-oriented training pipeline.

## Baseline

Run:

    python baseline.py

The baseline intentionally keeps several concerns together so we can identify what should be separated:
- data loading
- model definition
- training
- evaluation
- checkpoint saving

During Session 1 we will start refactoring this into a cleaner structure.

## Requirements

Python 3.10+ recommended.

    pip install -r requirements.txt
