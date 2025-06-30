"""
Setup script for Weights & Biases integration.
This script helps configure wandb for the segmentation experiments.
"""

import subprocess
import sys
import os

def check_wandb_installation():
    """Check if wandb is installed"""
    try:
        import wandb
        print("✅ wandb is already installed")
        return True
    except ImportError:
        print("❌ wandb is not installed")
        return False

def install_wandb():
    """Install wandb using pip"""
    print("📦 Installing wandb...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
        print("✅ wandb installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install wandb")
        return False

def setup_wandb_login():
    """Setup wandb login"""
    print("\n🔐 Setting up wandb login...")
    print("You need to log in to wandb to track your experiments.")
    print("If you don't have an account, visit: https://wandb.ai/signup")

    try:
        subprocess.check_call(["wandb", "login"])
        print("✅ wandb login successful")
        return True
    except subprocess.CalledProcessError:
        print("❌ wandb login failed")
        return False
    except FileNotFoundError:
        print("❌ wandb command not found. Try restarting your terminal.")
        return False

def test_wandb():
    """Test wandb functionality"""
    print("\n🧪 Testing wandb functionality...")
    try:
        import wandb

        wandb.init(
            project="wandb_test",
            name="setup_test",
            mode="disabled"
        )

        wandb.log({"test_metric": 0.5})
        wandb.finish()

        print("✅ wandb test successful")
        return True
    except Exception as e:
        print(f"❌ wandb test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Weights & Biases for Segmentation Experiments")
    print("=" * 60)

    if not check_wandb_installation():
        if not install_wandb():
            print("\n❌ Setup failed: Could not install wandb")
            return False

    if not setup_wandb_login():
        print("\n❌ Setup failed: Could not login to wandb")
        return False

    if not test_wandb():
        print("\n❌ Setup failed: wandb test failed")
        return False

    print("\n" + "=" * 60)
    print("🎉 wandb setup completed successfully!")
    print("\nYou can now run your experiments with:")
    print("  python train.py --config config/no_augmentation.yaml")
    print("  python train.py --config config/light_geometric.yaml")
    print("  python train.py --config config/color_light_geometric.yaml")
    print("  python train.py --config config/heavy_augmentation.yaml")
    print("\nOr run all experiments at once with:")
    print("  python run_augmentation_experiments.py")
    print(f"\n🔗 View your results at: https://wandb.ai")

    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)