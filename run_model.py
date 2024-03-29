# Imports
import os
from utils import import_model, import_system, import_task, parse_configurations
from visualize import plot_overall_results, plot_timeline

# Run performance model for given configurations
def main():
    args = parse_configurations()
    model = import_model(args.model_cfg_file)
    system = import_system(args.system_cfg_file)
    task = import_task(model, system, args.task_cfg_file)

    if task.type == 'pretrain':
        computation_stream, communication_stream = task.build_pretrain(model, system)
    elif task.type == 'inference':
        computation_stream, communication_stream = task.build_inference(model, system)
    if task.type == 'finetune':
        computation_stream, communication_stream = task.build_finetune(model, system)

    if not os.path.exists('figures'):
        os.mkdir('figures')

    plot_overall_results(task, args.figures_dir)
    plot_timeline(computation_stream, communication_stream, args.figures_dir)

if __name__ == "__main__":
    main()