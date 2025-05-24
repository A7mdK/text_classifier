from data.processor import DatasetProcessor
from models.trainer import ModelTrainer
from config import MODELS, DATASETS
import json
from datetime import datetime
from config import TRAINING_PRESETS, RESULT_CONFIG
from models.visualizer import ResultVisualizer

def run_experiment(preset: str = "default", experiment_name: str = None, inspect=False, quick_test=False, save_results=True, custom_args: dict = None):
    """Full experiment pipeline"""
    results = {}

    # Get base config
    experiment_args = TRAINING_PRESETS["quick_test" if quick_test else preset].copy()

    # Apply custom overrides
    if custom_args:
        experiment_args.update(custom_args)
    
    for ds_name in DATASETS:
        print(f"\n🔧 Processing dataset: {ds_name}")
        dataset = DatasetProcessor.load_dataset(ds_name, quick_test)
        results[ds_name] = {}
        
        for model in MODELS:
            print(f"\n🚀 Training {model} on {ds_name}")
            try:
                metrics = ModelTrainer.train(model, dataset, inspect=inspect, **experiment_args)
                results[ds_name][model] = metrics
                print(f"✅ {model}: Acc={metrics['eval_accuracy']:.2f}, F1={metrics['eval_f1']:.2f}")
            except Exception as e:
                print(f"❌ Failed {model}: {str(e)}")
                results[ds_name][model] = {"error": str(e)}
    
    # Save and visualize results
    if save_results:
        if not experiment_name:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ResultVisualizer.save_results(results, experiment_name)
    
    return results

if __name__ == "__main__":
    run_experiment(preset="default", inspect=True)