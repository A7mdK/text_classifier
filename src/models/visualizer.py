import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from config import RESULT_CONFIG

class ResultVisualizer:
    
    @staticmethod
    def save_results(results: dict, experiment_name: str = None):
        """Save results in multiple formats"""
        if not experiment_name:
            experiment_name = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        os.makedirs(RESULT_CONFIG["save_dir"], exist_ok=True)
        base_path = f"{RESULT_CONFIG['save_dir']}/{experiment_name}"
        
        # 1. Save raw JSON
        with open(f"{base_path}.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # 2. Save figure
        if "plot" in RESULT_CONFIG["formats"]:
            ResultVisualizer._save_plots(results, base_path)
        
        # Print console summary
        ResultVisualizer._print_console_summary(results)


    @staticmethod
    def _save_plots(results: dict, base_path: str):
        """Generate plots directly from raw results"""
        
        # Accuracy plot
        ResultVisualizer._create_metric_plot(
            results, "accuracy", f"{base_path}_accuracy.png"
        )
        
        # F1 plot
        ResultVisualizer._create_metric_plot(
            results, "f1", f"{base_path}_f1.png"
        )

    @staticmethod
    def _create_metric_plot(results: dict, metric: str, save_path: str):
        """Create a single metric plot"""
        fig, ax = plt.subplots(figsize=(12, 6))

        width = 0.15
        dataset_count = len(results)
        
        for dataset_idx, (dataset, models) in enumerate(results.items()):
            model_names = []
            metric_values = []

            for model, metrics in models.items():
                model_names.append(model)
                val = metrics.get(f"eval_{metric}")
                metric_values.append(val)

            # Shift each dataset group for grouped bars
            bar_positions = [i + dataset_idx * width for i in range(len(model_names))]

            ax.bar(bar_positions, metric_values, width=width, label=dataset)

        total_width = width * dataset_count
        ax.set_xticks([i + total_width / 2 - width/2 for i in range(len(model_names))])
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_title(f"{metric.upper()} Comparison")
        ax.set_ylabel(metric)
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


    @staticmethod
    def _print_console_summary(results: dict):
        """Print formatted results to console"""
        print("\n=== Experiment Summary ===")
        for dataset, models in results.items():
            print(f"\nDataset: {dataset}")
            print("-"*40)
            for model, metrics in models.items():
                print(
                    f"{model[:20]:<20} | "
                    f"Acc: {metrics.get('eval_accuracy', 0):.3f} | "
                    f"F1: {metrics.get('eval_f1', 0):.3f}"
                )