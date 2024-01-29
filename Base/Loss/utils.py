import torch
import torch.nn.functional as F


def cross_entropy(y_pre, target, ignore_index=None):
    # 如果维度不对的话
    if not ignore_index is None:
        mask = target != ignore_index
    if len(y_pre.size()) != len(target.size()):
        target = F.one_hot(target.long(), num_classes=y_pre.size()[1])
        target = target.permute(0, 3, 1, 2)
    if ignore_index is None:
        # return -(target * torch.log(y_pre + 1e-8) + (1 - target) * torch.log(1 - y_pre + 1e-8))
        return -(target * torch.log(y_pre + 1e-8))
    if mask.size() != target.size():
        mask = torch.unsqueeze(mask, 1)
    return mask * (-(target * torch.log(y_pre + 1e-8) + (1 - target) * torch.log(1 - y_pre + 1e-8)))
    # return mask * (-(target * torch.log(y_pre + 1e-8)))


if __name__ == "__main__":
    print(torch.log(torch.Tensor([1e-8])))
    pred = torch.randn(1, 100)
    pred_soft = torch.softmax(pred + 1e-8, dim=1)
    target = torch.randint(0, 99, (1,)).to(torch.int64)
    import torch.nn.functional as F
    import torch.nn as nn

    target_one_hot = F.one_hot(target, num_classes=100)
    print(cross_entropy(pred_soft, target_one_hot).sum(dim=1).mean())
    print(nn.NLLLoss()(torch.log(pred_soft), target))
    print(nn.CrossEntropyLoss()(pred, target.to(torch.long)))
