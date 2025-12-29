"""
Production runner for Continual Learning Pipeline.

This script checks model performance and drift, and triggers retraining if needed.
Designed to be run periodically (e.g., daily or weekly via cron).

Usage:
    # Check last 7 days of production data
    python -m src.continual_learning.run_continual_learning --days-back 7

    # Check with specific end date
    python -m src.continual_learning.run_continual_learning --days-back 7 --end-date 2025-01-15

    # Dry run (don't deploy even if new model is better)
    python -m src.continual_learning.run_continual_learning --days-back 7 --dry-run

    # Force retraining regardless of checks
    python -m src.continual_learning.run_continual_learning --force-retrain
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from src.continual_learning.retraining_pipeline import ContinualLearningPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/continual_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def save_check_report(report: Dict[str, Any], output_dir: Path):
    """Save continual learning check report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"cl_report_{timestamp}.json"

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to {report_path}")
    return report_path


def run_continual_learning_check(
    days_back: int = 7,
    end_date: str = None,
    dry_run: bool = False,
    force_retrain: bool = False,
    reference_days: int = 30
) -> Dict[str, Any]:
    """
    Run continual learning check.

    Args:
        days_back: Number of days of production data to check
        end_date: End date for data (default: today)
        dry_run: If True, don't deploy even if new model is better
        force_retrain: If True, force retraining regardless of checks
        reference_days: Number of days for reference data (for drift detection)

    Returns:
        Report dictionary with check results
    """
    logger.info("="*80)
    logger.info("Starting Continual Learning Check")
    logger.info("="*80)

    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = ContinualLearningPipeline()

    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'days_back': days_back,
            'end_date': end_date,
            'dry_run': dry_run,
            'force_retrain': force_retrain,
            'reference_days': reference_days
        },
        'actions_taken': []
    }

    try:
        # Step 1: Load production data
        logger.info(f"Loading production data (last {days_back} days, end_date={end_date})...")
        production_data = pipeline.load_production_data(days_back=days_back, end_date=end_date)
        logger.info(f"Loaded {len(production_data):,} records")
        report['production_data_records'] = len(production_data)

        if len(production_data) == 0:
            logger.warning("No production data found! Exiting.")
            report['status'] = 'no_data'
            return report

        # Step 2: Check performance
        logger.info("\n--- Performance Check ---")
        performance_metrics = pipeline.check_performance(production_data)
        report['performance_metrics'] = performance_metrics

        logger.info("Performance Metrics:")
        for metric, value in performance_metrics.items():
            logger.info(f"  {metric}: {value}")

        # Step 3: Check drift
        logger.info("\n--- Drift Detection ---")
        logger.info(f"Loading reference data (last {reference_days} days from training period)...")
        reference_data = pipeline.load_production_data(days_back=reference_days, end_date="2024-12-31")
        logger.info(f"Reference data: {len(reference_data):,} records")

        drift_results = pipeline.check_drift(production_data, reference_data)
        report['drift_results'] = drift_results

        logger.info(f"Drift Detection Results:")
        logger.info(f"  Features checked: {len(drift_results.get('features_checked', []))}")
        logger.info(f"  Drift detected: {len(drift_results.get('drift_detected_features', []))}")
        logger.info(f"  Drifted features: {drift_results.get('drift_detected_features', [])}")

        # Step 4: Decide on retraining
        logger.info("\n--- Retraining Decision ---")

        if force_retrain:
            should_retrain = True
            reasons = ["Force retrain flag enabled"]
            logger.info("🚨 FORCE RETRAINING ENABLED")
        else:
            should_retrain, reasons = pipeline.should_retrain(performance_metrics, drift_results)

        report['should_retrain'] = should_retrain
        report['retrain_reasons'] = reasons

        if should_retrain:
            logger.info("🚨 RETRAINING TRIGGERED")
            logger.info("Reasons:")
            for reason in reasons:
                logger.info(f"  • {reason}")
            report['actions_taken'].append('retrain_triggered')

            # Step 5: Retrain model
            logger.info("\n--- Model Retraining ---")
            logger.info("Training new model...")

            new_model_path, new_model_version, train_metrics = pipeline.retrain_model(
                training_data=None,
                training_end_date=end_date
            )

            logger.info(f"✓ New model trained: Version {new_model_version}")
            logger.info("Training Metrics:")
            for metric, value in train_metrics.items():
                logger.info(f"  {metric}: {value}")

            report['new_model_version'] = new_model_version
            report['new_model_path'] = str(new_model_path)
            report['train_metrics'] = train_metrics
            report['actions_taken'].append('model_retrained')

            # Step 6: A/B Testing
            logger.info("\n--- A/B Testing ---")
            current_model_path = pipeline.config.model_path
            comparison = pipeline.compare_models(current_model_path, new_model_path, production_data)

            report['model_comparison'] = comparison

            logger.info("Model Comparison:")
            logger.info(f"  Current Model MAE: {comparison['current_model']['mae']:.4f}")
            logger.info(f"  New Model MAE: {comparison['new_model']['mae']:.4f}")
            logger.info(f"  Winner: {comparison['winner']}")

            # Step 7: Deploy (if not dry run)
            if comparison['winner'] == 'new':
                logger.info("🏆 NEW MODEL WINS!")

                if dry_run:
                    logger.info("⚠️  DRY RUN MODE - Skipping deployment")
                    report['actions_taken'].append('deployment_skipped_dry_run')
                    report['status'] = 'success_dry_run'
                else:
                    logger.info("Deploying new model...")
                    success = pipeline.deploy_model(new_model_path, new_model_version)

                    if success:
                        logger.info(f"✓ Model v{new_model_version} deployed successfully!")
                        report['actions_taken'].append('model_deployed')
                        report['deployed_version'] = new_model_version
                        report['status'] = 'success_deployed'
                    else:
                        logger.error("✗ Deployment failed!")
                        report['actions_taken'].append('deployment_failed')
                        report['status'] = 'failure_deployment'
            else:
                logger.info("⚠️  Current model is still better. No deployment.")
                report['actions_taken'].append('deployment_skipped_worse_performance')
                report['status'] = 'success_no_deployment'

        else:
            logger.info("✅ NO RETRAINING NEEDED")
            logger.info("Model performance and data distribution are stable.")
            report['status'] = 'success_no_retrain'

        logger.info("\n" + "="*80)
        logger.info("Continual Learning Check Complete")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Error during continual learning check: {e}", exc_info=True)
        report['status'] = 'error'
        report['error'] = str(e)

    return report


def main():
    parser = argparse.ArgumentParser(
        description='Run Continual Learning Check',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check last 7 days
  python -m src.continual_learning.run_continual_learning --days-back 7

  # Check with specific date
  python -m src.continual_learning.run_continual_learning --days-back 7 --end-date 2025-01-15

  # Dry run (don't deploy)
  python -m src.continual_learning.run_continual_learning --days-back 7 --dry-run

  # Force retrain
  python -m src.continual_learning.run_continual_learning --force-retrain
        """
    )

    parser.add_argument(
        '--days-back',
        type=int,
        default=7,
        help='Number of days of production data to check (default: 7)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for data in YYYY-MM-DD format (default: today)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run mode - check and retrain but don\'t deploy'
    )

    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force retraining regardless of checks'
    )

    parser.add_argument(
        '--reference-days',
        type=int,
        default=30,
        help='Number of days for reference data (default: 30)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='monitoring/continual_learning_reports',
        help='Directory to save reports (default: monitoring/continual_learning_reports)'
    )

    args = parser.parse_args()

    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)

    # Run continual learning check
    report = run_continual_learning_check(
        days_back=args.days_back,
        end_date=args.end_date,
        dry_run=args.dry_run,
        force_retrain=args.force_retrain,
        reference_days=args.reference_days
    )

    # Save report
    output_dir = Path(args.output_dir)
    report_path = save_check_report(report, output_dir)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Status: {report['status']}")
    print(f"Actions taken: {', '.join(report['actions_taken']) if report['actions_taken'] else 'None'}")
    print(f"Report saved to: {report_path}")
    print("="*80)

    # Exit with appropriate code
    if report['status'].startswith('success') or report['status'] == 'success_no_retrain':
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()
