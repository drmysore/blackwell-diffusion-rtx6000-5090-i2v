# Create a test file: test_env.py
import sys

print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")

# Test your packages
try:
    import torch

    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")
