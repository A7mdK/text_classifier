from experiment import run_experiment
import torch

print(torch.cuda.is_available())
print(torch.__version__) 

# Quick test with very a small portion of dataset
results = run_experiment(quick_test=True, save_results=True, inspect=True)
print(results)