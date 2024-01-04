# des: 传入的常量需不需要同device
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from .utils import cross_entropy


class MyFocalLoss(_Loss):
    def __init__(self, gamma=2, alpha=None, ignore_index=None, from_logits=False, num_classes=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.from_logits = from_logits
        self.num_classes = num_classes

    def forward(self, y_pred: torch.Tensor, target: torch.Tensor, is_one_hot=True):
        if self.from_logits:
            y_pred = torch.softmax(y_pred + 1e-4, dim=1)
        if not is_one_hot:
            # 构造好一个target
            target = F.one_hot(target.to(torch.int64), num_classes=self.num_classes)
        target = target.type_as(y_pred)
        ce_loss = self.focal_loss_with_logit(y_pred, target)

        return ce_loss

    def focal_loss_with_logit(self, y_pred, target):
        # 这边的focal loss 用CE似乎不太好，是不是可以直接使用mse
        log_pt = cross_entropy(y_pred, target, self.ignore_index)
        pt = torch.exp(-log_pt)
        loss = (1 - pt).pow(self.gamma) * log_pt
        if not self.alpha is None:
            loss *= self.alpha * target + (1 - self.alpha) * (1 - target)
        return loss.sum(dim=1).mean()


if __name__ == "__main__":
    t = torch.eye(3)
    p = torch.randn((3, 3)).sigmoid()
    t[0, 0] = 3
    print(t, p)
    print(MyFocalLoss(2, ignore_index=3)(p, t))
    print(cross_entropy(p, t, ignore_index=3))
