from models.model import Model

class DLRM(Model):
    # Get bytes from sparse embedding lookups
    def get_lookup_bytes(self):
        lookup_bytes = self.num_tables * self.pooling_size * self.emb_dim * self.bytes_per_emb_param
        return lookup_bytes
    
    # Get number of FLOPs per sample.
    def get_num_flops(self):
        mlp_flops_layer = 2 * self.mlp_layer_params
        mlp_flops_total = mlp_flops_layer * self.num_mlp_layers
        return mlp_flops_layer, mlp_flops_total
    
    # Get number of parameters
    def get_num_params(self):
        mlp_params = self.num_mlp_layers * self.mlp_layer_params
        emb_params = self.num_tables * self.entries_per_table * self.emb_dim
        total_params = mlp_params + emb_params
        return total_params, mlp_params, emb_params

    # Print model summary statistics
    def print_summary_stats(self):
        total_params_b = self.total_params / 1e9
        perc_dense_params = (self.mlp_params / self.total_params) * 100.
        perc_sparse_params = 100. - perc_dense_params

        dense_size_gb = (self.mlp_params * self.bytes_per_nonemb_param) / 1e9
        sparse_size_gb = (self.emb_params * self.bytes_per_emb_param) / 1e9
        total_size_gb = dense_size_gb + sparse_size_gb

        mflops_layer = self.layer_flops / 1e6
        mflops_total = self.total_flops / 1e6

        lookup_bytes_mb = self.lookup_bytes / 1e6

        print('**************************************************')
        super().print_summary_stats()
        print('Parameters: {:.2f} B ({:.2f}% dense, {:.2f}% sparse).'.format(total_params_b, perc_dense_params, perc_sparse_params))
        print('Size: {:.2f} GB ({:.2f} GB dense, {:.2f} GB sparse).'.format(total_size_gb, dense_size_gb, sparse_size_gb))
        print('FLOPs: {:.2f} MFLOPs ({:.2f} MFLOPs per MLP layer) per sample.'.format(mflops_total, mflops_layer))
        print('Lookup Bytes: {:.2f} MB per sample.'.format(lookup_bytes_mb))
        print('**************************************************')

    def __init__(
        self, 
        model_cfg
    ):
        super().__init__(model_cfg)
        self.num_bot_mlp_layers = model_cfg['num_bot_mlp_layers']
        self.num_top_mlp_layers = model_cfg['num_top_mlp_layers']
        self.num_mlp_layers = self.num_bot_mlp_layers + self.num_top_mlp_layers
        self.mlp_dim = model_cfg['mlp_dim']

        self.num_tables = model_cfg['num_tables']
        self.entries_per_table = model_cfg['entries_per_table']
        self.emb_dim = model_cfg['emb_dim']
        self.pooling_size = model_cfg['pooling_size']

        self.mlp_layer_params = self.mlp_dim * self.mlp_dim
        self.total_params, self.mlp_params, self.emb_params = self.get_num_params()
        self.layer_flops, self.total_flops = self.get_num_flops()
        self.lookup_bytes = self.get_lookup_bytes()

        self.print_summary_stats()