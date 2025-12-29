"""
Continual Learning module for NYC Taxi Demand Prediction.

This module provides automated model monitoring, drift detection, and retraining
capabilities to maintain model performance in production environments.

Features:
- Performance monitoring with threshold-based triggers
- Statistical drift detection (KS test, PSI)
- Automated model retraining with sliding window
- A/B testing for safe model comparison
- Model versioning and registry
- Automated deployment with backup

Components:
- ContinualLearningPipeline: Main orchestrator for continual learning workflow
- ModelRegistry: Version control and metadata tracking for models
- Performance monitoring: MAE-based threshold detection
- Drift detection: Integration with DriftDetector for statistical tests

Usage:
    # Run weekly continual learning check
    python -m src.continual_learning.run_continual_learning --days-back 7

    # Simulate multiple weeks
    python -m src.continual_learning.demo_continual_learning --weeks 4

    # Force retrain
    python -m src.continual_learning.run_continual_learning --force-retrain
"""

from src.continual_learning.retraining_pipeline import ContinualLearningPipeline

__all__ = ['ContinualLearningPipeline']
