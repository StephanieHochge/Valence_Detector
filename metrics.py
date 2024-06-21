import torch


def torch_PCC(ground_truth: torch.Tensor, predictions: torch.Tensor) -> float:
    """
    Calculate the pearson correlation coefficient (PCC) for the ground truth and the predictions.
    Function adapted from https://github.com/face-analysis/emonet/blob/master/emonet/metrics.py

    :param ground_truth: The ground truth values.
    :param predictions: The predicted values.
    :return: The computed PCC value.
    """
    std_pred = torch.std(predictions)
    std_gt = torch.std(ground_truth)

    if torch.equal(ground_truth, predictions):
        return 1

    if (std_pred == 0) | (std_gt == 0):
        return 0

    matrix = torch.stack((ground_truth, predictions), dim=0)
    corr = torch.corrcoef(matrix)[0, 1]

    return corr.item()


def torch_CCC(ground_truth: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    """
    Calculate the concordance correlation coefficient (CCC) for the ground truth and the predictions.
    Function adapted from https://github.com/face-analysis/emonet/blob/master/emonet/metrics.py

    :param ground_truth: The ground truth values.
    :param predictions: The predicted values.
    :return: The computed CCC value.
    """
    mean_pred = torch.mean(predictions)
    mean_gt = torch.mean(ground_truth)

    std_pred = torch.std(predictions)
    std_gt = torch.std(ground_truth)

    if torch.equal(ground_truth, predictions):
        return torch.Tensor([1])

    if (std_pred == 0) | (std_gt == 0):
        return torch.Tensor([0])

    pearson = torch_PCC(ground_truth, predictions)
    ccc = 2.0 * pearson * std_pred * std_gt / (std_pred ** 2 + std_gt ** 2 + (mean_pred - mean_gt) ** 2)
    return ccc

