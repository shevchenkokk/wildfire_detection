import os
import time
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir_base="runs/experiment", experiment_name=None, hparams=None):
        """
        Initializes a TensorBoard logger.

        Args:
            log_dir_base (str): Base directory for all runs.
            experiment_name (str, optional): Name for the current experiment. 
                                         If None, a timestamp will be used.
            hparams (dict, optional): Hyperparameters to log.
        """
        current_time = time.strftime("%Y%m%d-%H%M%S")
        if experiment_name:
            self.log_dir = os.path.join(log_dir_base, experiment_name, current_time)
        else:
            self.log_dir = os.path.join(log_dir_base, current_time)
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"TensorBoard logs will be saved to: {self.log_dir}")

        if hparams:
            self.log_hparams(hparams)

    def log_scalar(self, tag, scalar_value, global_step):
        """Logs a scalar value."""
        self.writer.add_scalar(tag, scalar_value, global_step)

    def log_hparams(self, hparams_dict, metrics_dict=None):
        """Logs hyperparameters. Optionally logs initial metrics alongside."""
        # For add_hparams, it's good practice to log some placeholder metric as well.
        if metrics_dict is None:
            metrics_dict = {} # metrics_dict should be a dict, e.g. {'hparam/accuracy': 0.1, 'hparam/loss': 10}
        self.writer.add_hparams(hparams_dict, metrics_dict)

    def log_model_graph(self, model, input_to_model=None, verbose=False):
        """
        Logs the model graph.
        Args:
            model (torch.nn.Module): The model to log.
            input_to_model (torch.Tensor or tuple of torch.Tensor, optional): 
                A sample input to the model needed to trace the graph. 
                For models with multiple inputs, provide a tuple.
            verbose (bool): Whether to print a verbose output of the graph tracing.
        """
        if input_to_model is None:
            print("Warning: input_to_model is None. Cannot log model graph.")
            return
        try:
            self.writer.add_graph(model, input_to_model, verbose=verbose)
            print("Model graph added to TensorBoard.")
        except Exception as e:
            print(f"Could not add model graph to TensorBoard: {e}")

    def log_image_grid(self, tag, image_grid, global_step):
        """Logs a grid of images."""
        self.writer.add_image(tag, image_grid, global_step)
        print(f"Logged image grid '{tag}' for step {global_step} to TensorBoard.")
    
    def log_text(self, tag, text_string, global_step):
        """Logs text."""
        self.writer.add_text(tag, text_string, global_step)

    def close(self):
        """Closes the SummaryWriter."""
        self.writer.close()
        print(f"TensorBoard writer closed. Logs are in: {self.log_dir}")