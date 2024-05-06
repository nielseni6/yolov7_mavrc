import torch
import torch.nn as nn

from torchattacks.attack import Attack


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, loss, metric, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True, augment=False):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.loss = loss
        self.metric = metric
        self.augment = augment
    
    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = self.loss
        metric = self.metric
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            _, train_out = self.model(adv_images, augment=self.augment)
            outputs = [x.float().requires_grad_(True) for x in train_out]
            
            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels, metric=metric)[0]  # sum(box, obj, cls)
            else:
                cost = loss(outputs, labels, metric=metric)[0]  # sum(box, obj, cls)
            # cost.requires_grad_(True)
            
            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, 
                retain_graph=True, 
                # retain_graph=True, 
                create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images, delta

class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, loss, metric, eps=8 / 255, augment=False):
        super().__init__("FGSM", model)
        self.eps = eps
        self.loss = loss
        self.metric = metric
        self.augment = augment
        
    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = labels # self.get_target_label(images, labels)
            
        loss = self.loss # nn.CrossEntropyLoss()
        metric = self.metric

        images.requires_grad = True
        # outputs = self.get_logits(images)
        _, train_out = self.model(images, augment=self.augment)
        outputs = [x.float().requires_grad_(True) for x in train_out]

        # Calculate loss
        if self.targeted:
            cost = loss(outputs, target_labels, metric=metric)[0]  # sum(box, obj, cls)
        else:
            cost = loss(outputs, labels, metric=metric)[0]  # sum(box, obj, cls)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, images, retain_graph=True, create_graph=False
        )[0]

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        delta = self.eps * grad.sign()

        return adv_images, delta
