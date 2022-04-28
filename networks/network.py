import torch.nn.functional as F
import torch.nn
import re

class Network:
    def __init__(self, model, layer_names, pool_factors=None):
        self.layer_names = layer_names
        self.pool_factors = pool_factors
        if isinstance(model, torch.nn.DataParallel):
            self.model = model.module
        else:
            self.model = model
        self.activations = dict()
        if pool_factors is None:
            pool_factors = dict()
            for layer_name in layer_names:
                pool_factors[layer_name] = 1
        if layer_names is not None:
            d = dict(self.model.named_modules())
            print('Will fetch activations from:')
            for layer_name in layer_names:
                if layer_name in d:
                    layer = self.getLayer(layer_name)
                    pool_factor = pool_factors[layer_name]
                    layer_rep = re.match('.+($|\n)', layer.__repr__())
                    print('{}, average pooled by {}:'.format(layer_name, pool_factor), layer_rep.group(0).strip())
                    # print('{}, average pooled by {}:'.format(layer_name, pool_factor))
                    layer.register_forward_hook(self.getActivation(layer_name, pool_factor))
                else:
                    print("Warning: Layer {} not found".format(layer_name))
            # for i, layer_name in enumerate(layer_names):
            #     pool_factor = pool_factors[i]
            #     if layer_name in d:
            #         layer = self.getLayer(layer_name)
            #         print('{}, average pooled by {}:'.format(layer_name, pool_factor), layer)
            #         layer.register_forward_hook(self.getActivation(layer_name, pool_factor))

    def __repr__(self):
        out = 'Layers {}\n'.format(self.layer_names)
        if self.pool_factors:
            out = '{}Pool factors {}\n'.format(out, list(self.pool_factors.values()))
        out = '{}'.format(self.model.__repr__())
        return out


    def getActivation(self, name, pool):
        def hook(module, input, output):
            layer_out = output.detach()
            if layer_out.dim() == 3 and pool > 1:
                layer_out_pool = F.avg_pool1d(layer_out, pool)
            elif layer_out.dim() == 4 and pool > 1:
                layer_out_pool = F.avg_pool2d(layer_out, pool)
            elif layer_out.dim() == 5 and pool > 1:
                layer_out_pool = F.avg_pool3d(layer_out, pool)
            else:
                layer_out_pool = layer_out
            self.activations[name] = layer_out_pool.view(output.size(0), -1)
            # self.activations[name] = layer_out
        return hook

    def __call__(self, data):
        # self.activations.clear()
        out = self.model(data)
        return out, self.activations

    def getLayer(self, layer_name):
        m = self.model
        sep = '.'
        attrs = layer_name.split(sep)
        for a in attrs:
            try:
                i = int(a)
                m = m[i]
            except ValueError:
                m = m.__getattr__(a)
        return m



