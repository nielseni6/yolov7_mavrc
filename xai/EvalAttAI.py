from plaus_functs import get_gradient
import torch

class EvalAttAI:
    def __init__(self, model, nsteps = 10, epsilon = 0.05, augment = False):
        self.model = model
        self.nsteps = nsteps
        self.epsilon = epsilon
        self.augment = augment
    
    def __init_attr__(self, attr_method = get_gradient, **kwargs):
        self.attr_method = attr_method
        self.attr_kwargs = kwargs

        
    def collect_stats(self, img, grad_wrt):
        # for key, value in self.attr_kwargs.items():
        #     self.key = value
        self.attr = self.attr_method(img.detach().requires_grad_(True), grad_wrt, **self.attr_kwargs)
        
        for i in range(self.nsteps):
            img += (self.epsilon * self.attr)
            with torch.no_grad():
                out, train_out = self.model(img, augment=self.augment)
                # add more stats here like map, etc.
                # stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
        return 
    
    def return_faithfulness_score(self):
        return