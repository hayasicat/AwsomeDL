import torch


def cross_entropy(y_pre, target, ignore_index=None):
    if ignore_index is None:
        # return -(target * torch.log(y_pre + 1e-8) + (1 - target) * torch.log(1 - y_pre + 1e-8))
        return -(target * torch.log(y_pre + 1e-8))
    mask = target != ignore_index
    return mask * (-(target * torch.log(y_pre + 1e-8) + (1 - target) * torch.log(1 - y_pre + 1e-8)))


if __name__ == "__main__":
    pred = torch.randn(1, 100)
    pred_soft = torch.softmax(pred + 1e-8, dim=1)
    target = torch.randint(0, 99, (1,)).to(torch.int64)
    import torch.nn.functional as F
    import torch.nn as nn

    target_one_hot = F.one_hot(target, num_classes=100)
    print(cross_entropy(pred_soft, target_one_hot).sum(dim=1).mean())
    print(nn.NLLLoss()(torch.log(pred_soft), target))
    print(nn.CrossEntropyLoss()(pred, target.to(torch.long)))
