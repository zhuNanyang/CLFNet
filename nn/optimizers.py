import torch




class Optimizer:
    def __init__(self, model_params, **kwargs):
        if model_params is not None and not hasattr(model_params, "__next__"):
            raise TypeError("model parameters should be a generator, rather than {}.".format(type(model_params)))
        self.model_params = model_params
        self.settings = kwargs

    def to_torch(self, model_params) -> "torch.optim.Optimizer":
        raise NotImplementedError

    @staticmethod
    def get_requires_grad_params(params):
        return [param for param in params if param.requires_grad]


class SGD(Optimizer):
    def __init__(self, lr=0.001, momentum=0.0, model_params=None):
        if not isinstance(lr, float):

            raise TypeError("learning rate has to be float.")
        super(SGD, self).__init__(model_params, lr=lr, momentum=momentum)

    def to_torch(self, model_params):
        if self.model_params is None:
            # careful! generator cannot be assigned.
            return torch.optim.SGD(self.get_requires_grad_params(model_params), **self.settings)
        else:
            return torch.optim.SGD(self.get_requires_grad_params(self.model_params), **self.settings)



class Adam(Optimizer):
    def __init__(self, lr=0.001, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8, amsgrad=False, model_params=None):
        if not isinstance(lr, float):
            raise TypeError("learning rate has to be float.")
        super(Adam, self).__init__(
            model_params, lr=lr, betas=betas, eps=eps, amsgrad=amsgrad, weight_decay=weight_decay
        )

    def to_torch(self, model_params):
        if self.model_params is None:
            # careful! generator cannot be assigned.

            return torch.optim.Adam(self.get_requires_grad_params(model_params), **self.settings)
        else:
            return torch.optim.Adam(self.get_requires_grad_params(self.model_params), **self.settings)



class AdamW(Optimizer):
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, model_params=None):

        super(AdamW, self).__init__(
            model_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )

    def to_torch(self, model_params):
        if self.model_params is None:

            # careful! generator cannot be assigned.
            return torch.optim.AdamW(self.get_requires_grad_params(model_params), **self.settings)
        else:
            return torch.optim.AdamW(self.get_requires_grad_params(self.model_params), **self.settings)
