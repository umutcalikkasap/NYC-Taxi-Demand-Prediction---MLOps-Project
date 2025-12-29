"""
Demo script to showcase the Continual Learning Pipeline.

This script simulates a weekly continual learning cycle:
1. Load current production model
2. Gather production data from last N days
3. Check for drift and performance degradation
4. Trigger retraining if needed
5. A/B test new model vs current model
6. Deploy winning model

Usage:
    python -m src.continual_learning.demo_continual_learning --weeks 4
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.continual_learning.retraining_pipeline import ContinualLearningPipeline

console = Console()


def print_header(text: str):
    """Print a formatted header."""
    console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
    console.print(f"[bold yellow]{text}[/bold yellow]")
    console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")


def print_metrics_table(metrics: dict, title: str = "Performance Metrics"):
    """Print metrics in a beautiful table."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", style="green", justify="right", width=20)

    for metric, value in metrics.items():
        if isinstance(value, float):
            if 'pct' in metric.lower() or 'percentage' in metric.lower():
                formatted_value = f"{value:.2f}%"
            elif value < 1:
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = f"{value:.2f}"
        else:
            formatted_value = str(value)
        table.add_row(metric, formatted_value)

    console.print(table)


def print_drift_summary(drift_results: dict):
    """Print drift detection summary."""
    table = Table(title="Drift Detection Summary", show_header=True, header_style="bold magenta")
    table.add_column("Feature", style="cyan", width=20)
    table.add_column("KS Statistic", style="yellow", justify="right", width=15)
    table.add_column("P-Value", style="yellow", justify="right", width=15)
    table.add_column("PSI", style="yellow", justify="right", width=15)
    table.add_column("Drift?", style="bold", justify="center", width=10)

    features_checked = drift_results.get('features_checked', [])
    drift_detected = drift_results.get('drift_detected_features', [])

    for feature in features_checked:
        if feature in drift_results.get('drift_scores', {}):
            scores = drift_results['drift_scores'][feature]
            ks_stat = scores.get('ks_statistic', 0)
            p_value = scores.get('p_value', 1)
            psi = scores.get('psi', 0)
            has_drift = feature in drift_detected

            drift_emoji = "🔴 YES" if has_drift else "✅ NO"
            if has_drift:
                style = "bold red"
            else:
                style = "green"

            table.add_row(
                feature,
                f"{ks_stat:.4f}",
                f"{p_value:.4f}",
                f"{psi:.4f}",
                f"[{style}]{drift_emoji}[/{style}]"
            )

    console.print(table)

    # Summary
    total_features = len(features_checked)
    drifted_features = len(drift_detected)
    drift_pct = (drifted_features / total_features * 100) if total_features > 0 else 0

    summary_text = f"[bold]Total Features:[/bold] {total_features} | "
    summary_text += f"[bold red]Drifted:[/bold red] {drifted_features} ({drift_pct:.1f}%)"
    console.print(Panel(summary_text, title="Summary", border_style="cyan"))


def simulate_weekly_check(pipeline: ContinualLearningPipeline, week_num: int, end_date: str):
    """Simulate a weekly continual learning check."""

    print_header(f"📅 Week {week_num} Check - {end_date}")

    # Step 1: Load production data
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Loading production data...", total=None)
        production_data = pipeline.load_production_data(days_back=7, end_date=end_date)
        progress.update(task, completed=True)

    console.print(f"[green]✓ Loaded {len(production_data):,} production records[/green]")

    # Step 2: Check performance
    console.print("\n[bold]Step 1: Performance Check[/bold]")
    performance_metrics = pipeline.check_performance(production_data)
    print_metrics_table(performance_metrics, "Current Performance Metrics")

    # Step 3: Check drift
    console.print("\n[bold]Step 2: Drift Detection[/bold]")
    reference_data = pipeline.load_production_data(days_back=30, end_date="2024-12-31")
    drift_results = pipeline.check_drift(production_data, reference_data)
    print_drift_summary(drift_results)

    # Step 4: Decide if retraining is needed
    console.print("\n[bold]Step 3: Retraining Decision[/bold]")
    should_retrain, reasons = pipeline.should_retrain(performance_metrics, drift_results)

    if should_retrain:
        console.print(f"[bold red]🚨 RETRAINING TRIGGERED[/bold red]")
        console.print(f"[yellow]Reasons:[/yellow]")
        for reason in reasons:
            console.print(f"  • {reason}")

        # Step 5: Retrain
        console.print("\n[bold]Step 4: Model Retraining[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Training new model...", total=None)

            # Use sliding window: last 60 days of training data + new production data
            training_end_date = end_date
            new_model_path, new_model_version, train_metrics = pipeline.retrain_model(
                training_data=None,  # Will load from parquet
                training_end_date=training_end_date
            )
            progress.update(task, completed=True)

        console.print(f"[green]✓ New model trained: Version {new_model_version}[/green]")
        print_metrics_table(train_metrics, f"New Model (v{new_model_version}) Training Metrics")

        # Step 6: A/B Testing
        console.print("\n[bold]Step 5: A/B Testing[/bold]")
        current_model_path = pipeline.config.model_path
        comparison = pipeline.compare_models(current_model_path, new_model_path, production_data)

        # Show comparison table
        table = Table(title="Model Comparison", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Current Model", style="yellow", justify="right", width=20)
        table.add_column("New Model", style="yellow", justify="right", width=20)
        table.add_column("Improvement", style="bold", justify="right", width=20)

        for metric in ['mae', 'rmse', 'r2', 'mape']:
            current_val = comparison['current_model'].get(metric, 0)
            new_val = comparison['new_model'].get(metric, 0)

            # Calculate improvement
            if metric in ['mae', 'rmse', 'mape']:
                improvement = ((current_val - new_val) / current_val * 100) if current_val > 0 else 0
                improvement_str = f"{improvement:+.2f}%"
                style = "green" if improvement > 0 else "red"
            else:  # r2
                improvement = ((new_val - current_val) / abs(current_val) * 100) if current_val != 0 else 0
                improvement_str = f"{improvement:+.2f}%"
                style = "green" if improvement > 0 else "red"

            table.add_row(
                metric.upper(),
                f"{current_val:.4f}",
                f"{new_val:.4f}",
                f"[{style}]{improvement_str}[/{style}]"
            )

        console.print(table)

        winner = comparison['winner']
        if winner == 'new':
            console.print(f"\n[bold green]🏆 NEW MODEL WINS! Deploying...[/bold green]")

            # Step 7: Deploy
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Deploying new model...", total=None)
                success = pipeline.deploy_model(new_model_path, new_model_version)
                progress.update(task, completed=True)

            if success:
                console.print(f"[bold green]✓ Model v{new_model_version} deployed successfully![/bold green]")
            else:
                console.print("[bold red]✗ Deployment failed![/bold red]")
        else:
            console.print(f"\n[bold yellow]⚠️  Current model is still better. No deployment.[/bold yellow]")

    else:
        console.print(f"[bold green]✅ NO RETRAINING NEEDED[/bold green]")
        console.print(f"[dim]Model performance and data distribution are stable.[/dim]")

    console.print("\n" + "─" * 80)

    return should_retrain


def main():
    parser = argparse.ArgumentParser(description='Continual Learning Pipeline Demo')
    parser.add_argument('--weeks', type=int, default=4,
                      help='Number of weeks to simulate (default: 4)')
    parser.add_argument('--start-date', type=str, default='2025-01-07',
                      help='Start date for simulation (YYYY-MM-DD)')
    args = parser.parse_args()

    # Print intro
    console.print(Panel.fit(
        "[bold cyan]Continual Learning Pipeline Demo[/bold cyan]\n\n"
        "This demo simulates weekly model monitoring and retraining.\n"
        f"Simulating {args.weeks} weeks of production monitoring...",
        border_style="cyan"
    ))

    # Initialize pipeline
    console.print("\n[bold]Initializing Continual Learning Pipeline...[/bold]")
    pipeline = ContinualLearningPipeline()
    console.print("[green]✓ Pipeline initialized[/green]")

    # Simulate weekly checks
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    retraining_count = 0

    for week in range(1, args.weeks + 1):
        # Calculate end date for this week
        end_date = start_date + timedelta(weeks=week)
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Run weekly check
        did_retrain = simulate_weekly_check(pipeline, week, end_date_str)
        if did_retrain:
            retraining_count += 1

    # Final summary
    print_header("📊 Simulation Summary")

    summary_table = Table(show_header=False, box=None)
    summary_table.add_column("Item", style="cyan bold", width=30)
    summary_table.add_column("Value", style="green bold", width=20)

    summary_table.add_row("Total Weeks Simulated", str(args.weeks))
    summary_table.add_row("Retraining Events", str(retraining_count))
    summary_table.add_row("Retraining Frequency", f"{retraining_count}/{args.weeks} weeks")

    # Get final model info
    registry = pipeline.model_registry.load_registry()
    if registry:
        latest = registry[-1]
        summary_table.add_row("Current Model Version", latest['version'])
        summary_table.add_row("Model MAE", f"{latest['performance'].get('mae', 0):.4f}")

    console.print(summary_table)

    console.print(f"\n[bold green]✓ Simulation complete![/bold green]")
    console.print(f"\n[dim]Model registry: {pipeline.model_registry.registry_path}[/dim]")


if __name__ == "__main__":
    main()
