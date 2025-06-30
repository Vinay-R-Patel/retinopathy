"""
Script to run all augmentation experiments sequentially.
This will train models with different augmentation strategies and log everything to wandb.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_experiment(config_name, description):
    """Run a single experiment with the given config"""
    print(f"\n{'=' * 60}")
    print(f"Starting experiment: {description}")
    print(f"Config: {config_name}")
    print(f"{'=' * 60}")

    config_path = f"config/{config_name}.yaml"

    if not Path(config_path).exists():
        print(f"ERROR: Config file {config_path} not found!")
        return False

    try:
        result = subprocess.run([
            sys.executable, "train.py",
            "--config", config_path
        ], check=True)

        print(f"\n✅ Experiment '{description}' completed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Experiment '{description}' failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ Experiment '{description}' interrupted by user")
        return False

def main():
    """Run all augmentation experiments"""
    experiments = [
        ("no_augmentation", "No Augmentation (Baseline)"),
        ("light_geometric", "Light Geometric Augmentation"),
        ("color_light_geometric", "Color + Light Geometric Augmentation"),
        ("heavy_augmentation", "Heavy Augmentation")
    ]

    print("🚀 Starting Augmentation Comparison Experiments")
    print(f"Total experiments to run: {len(experiments)}")
    print("\nMake sure you have:")
    print("1. wandb installed (pip install wandb)")
    print("2. wandb login completed (wandb login)")
    print("3. Sufficient disk space for outputs")

    input("\nPress Enter to continue or Ctrl+C to cancel...")

    successful_experiments = 0
    failed_experiments = 0
    start_time = time.time()

    for i, (config_name, description) in enumerate(experiments, 1):
        print(f"\n🔄 Running experiment {i}/{len(experiments)}")

        success = run_experiment(config_name, description)

        if success:
            successful_experiments += 1
        else:
            failed_experiments += 1

        if i < len(experiments):
            print("\n⏱️ Waiting 30 seconds before next experiment...")
            time.sleep(30)

    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)

    print(f"\n{'=' * 60}")
    print("📊 EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")
    print(f"✅ Successful experiments: {successful_experiments}")
    print(f"❌ Failed experiments: {failed_experiments}")
    print(f"⏱️ Total time: {hours}h {minutes}m")
    print(f"\n🔗 Check your results at: https://wandb.ai")
    print("📊 Look for project: multiclass_segmentation_augmentation_test")

if __name__ == "__main__":
    main()