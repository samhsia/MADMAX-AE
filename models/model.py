# Model Architecture
class Model:
    # Print model summary statistics
    def print_summary_stats(self):
        print('Model Name: {}'.format(self.name))

    def __init__(
        self, 
        model_cfg
    ):
        self.name = model_cfg['name']
        self.type = model_cfg['type']
        self.bytes_per_nonemb_param = model_cfg['bytes_per_nonemb_param']
        self.bytes_per_emb_param = model_cfg['bytes_per_emb_param']