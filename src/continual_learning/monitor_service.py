"""
Background monitoring service for Continual Learning.

This service runs as a background process and periodically checks:
1. Model performance on recent production data
2. Data drift detection
3. Triggers retraining when needed

Can be run as a systemd service, cron job, or standalone daemon.

Usage:
    # Run as standalone daemon (checks every 24 hours)
    python -m src.continual_learning.monitor_service --interval 86400

    # Run once and exit
    python -m src.continual_learning.monitor_service --run-once

    # Custom check interval (in seconds)
    python -m src.continual_learning.monitor_service --interval 3600  # 1 hour
"""

import argparse
import time
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.continual_learning.run_continual_learning import run_continual_learning_check

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/continual_learning_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ContinualLearningMonitor:
    """
    Background service for continual learning monitoring.
    """

    def __init__(self, check_interval: int = 86400, days_back: int = 7):
        """
        Initialize the monitor.

        Args:
            check_interval: Time in seconds between checks (default: 86400 = 24h)
            days_back: Number of days of production data to check
        """
        self.check_interval = check_interval
        self.days_back = days_back
        self.running = False
        self.last_check_time: Optional[datetime] = None

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("="*80)
        logger.info("Continual Learning Monitor Service Initialized")
        logger.info("="*80)
        logger.info(f"Check interval: {check_interval} seconds ({check_interval/3600:.1f} hours)")
        logger.info(f"Days back: {days_back}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"\nReceived signal {signum}. Shutting down gracefully...")
        self.running = False

    def run_check(self) -> bool:
        """
        Run a single continual learning check.

        Returns:
            True if check completed successfully, False otherwise
        """
        logger.info("\n" + "="*80)
        logger.info(f"Starting Continual Learning Check - {datetime.now()}")
        logger.info("="*80)

        try:
            # Run continual learning check
            report = run_continual_learning_check(
                days_back=self.days_back,
                end_date=None,  # Use current date
                dry_run=False,  # Deploy if new model is better
                force_retrain=False
            )

            # Log summary
            status = report.get('status', 'unknown')
            actions = report.get('actions_taken', [])

            logger.info("\n" + "-"*80)
            logger.info("CHECK SUMMARY")
            logger.info("-"*80)
            logger.info(f"Status: {status}")
            logger.info(f"Actions: {', '.join(actions) if actions else 'None'}")

            if 'model_deployed' in actions:
                logger.info("✓ New model deployed successfully!")
            elif 'retrain_triggered' in actions:
                logger.info("Model retrained but not deployed (not better than current)")
            else:
                logger.info("No retraining needed - model is stable")

            logger.info("-"*80)

            self.last_check_time = datetime.now()
            return True

        except Exception as e:
            logger.error(f"Error during continual learning check: {e}", exc_info=True)
            return False

    def run_daemon(self):
        """
        Run as a daemon service with periodic checks.
        """
        logger.info("\n" + "="*80)
        logger.info("Starting Continual Learning Monitor Daemon")
        logger.info("="*80)
        logger.info(f"Next check will run in {self.check_interval} seconds")
        logger.info("Press Ctrl+C to stop")
        logger.info("="*80 + "\n")

        self.running = True
        check_count = 0

        while self.running:
            try:
                check_count += 1
                logger.info(f"\n>>> Check #{check_count} <<<")

                # Run the check
                success = self.run_check()

                if success:
                    logger.info(f"✓ Check #{check_count} completed successfully")
                else:
                    logger.warning(f"⚠ Check #{check_count} completed with errors")

                # Wait for next check
                if self.running:
                    logger.info(f"\nSleeping for {self.check_interval} seconds until next check...")
                    logger.info(f"Next check scheduled at: {datetime.now()}")

                    # Sleep in small intervals to allow for graceful shutdown
                    sleep_remaining = self.check_interval
                    while sleep_remaining > 0 and self.running:
                        sleep_time = min(sleep_remaining, 10)  # Check every 10 seconds
                        time.sleep(sleep_time)
                        sleep_remaining -= sleep_time

            except Exception as e:
                logger.error(f"Unexpected error in daemon loop: {e}", exc_info=True)
                logger.info("Continuing to next check...")

                # Brief wait before retry
                if self.running:
                    time.sleep(60)

        logger.info("\n" + "="*80)
        logger.info("Continual Learning Monitor Daemon Stopped")
        logger.info(f"Total checks completed: {check_count}")
        logger.info("="*80)

    def run_once(self):
        """Run a single check and exit."""
        logger.info("Running single continual learning check...")
        success = self.run_check()

        if success:
            logger.info("✓ Check completed successfully")
            sys.exit(0)
        else:
            logger.error("✗ Check failed")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Continual Learning Background Monitor Service',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run as daemon with default 24h interval
  python -m src.continual_learning.monitor_service

  # Run as daemon with 1h interval
  python -m src.continual_learning.monitor_service --interval 3600

  # Run single check and exit
  python -m src.continual_learning.monitor_service --run-once

  # Custom configuration
  python -m src.continual_learning.monitor_service --interval 7200 --days-back 14

Systemd Service Example:
  [Unit]
  Description=NYC Taxi Continual Learning Monitor
  After=network.target

  [Service]
  Type=simple
  User=ubuntu
  WorkingDirectory=/path/to/project
  ExecStart=/path/to/venv/bin/python -m src.continual_learning.monitor_service
  Restart=always
  RestartSec=10

  [Install]
  WantedBy=multi-user.target

Cron Job Example:
  # Run check every day at 2 AM
  0 2 * * * cd /path/to/project && /path/to/venv/bin/python -m src.continual_learning.monitor_service --run-once
        """
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=86400,
        help='Check interval in seconds (default: 86400 = 24 hours)'
    )

    parser.add_argument(
        '--days-back',
        type=int,
        default=7,
        help='Number of days of production data to check (default: 7)'
    )

    parser.add_argument(
        '--run-once',
        action='store_true',
        help='Run a single check and exit (instead of daemon mode)'
    )

    args = parser.parse_args()

    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)

    # Initialize monitor
    monitor = ContinualLearningMonitor(
        check_interval=args.interval,
        days_back=args.days_back
    )

    # Run
    if args.run_once:
        monitor.run_once()
    else:
        monitor.run_daemon()


if __name__ == "__main__":
    main()
