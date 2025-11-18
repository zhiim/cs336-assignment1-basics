import torch


def cross_entropy(pred: torch.Tensor, target: torch.Tensor):
    num_class = pred.size(-1)

    # subtract the largest element, it do not change sofmax proability
    # to avoid overflow of exp()
    pred_max, _ = torch.max(pred, dim=-1, keepdim=True)
    pred = pred - pred_max

    # do not apply log() to softmax output
    # to avoid log() on 0 value
    pred_ = torch.log(torch.sum(torch.exp(pred), dim=-1, keepdim=True)) - pred

    target_one_hot = torch.eye(num_class)[target]

    result = torch.mean(pred_ * target_one_hot) * num_class

    return result
