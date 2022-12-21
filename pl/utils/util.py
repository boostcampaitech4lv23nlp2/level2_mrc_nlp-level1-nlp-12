import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from datasets import load_metric
import logging
logger = logging.getLogger(__name__)

def compute_metrics(pred):
    metric = load_metric("squad")
    return metric.compute(predictions=pred.predictions, references=pred.label_ids)

# def check_no_error(
#     data_args: DataTrainingArguments,
#     training_args: TrainingArguments,
#     datasets: DatasetDict,
#     tokenizer,
# ) -> Tuple[Any, int]:

#     # last checkpoint 찾기.
#     last_checkpoint = None
#     if (
#         os.path.isdir(training_args.output_dir)
#         and training_args.do_train
#         and not training_args.overwrite_output_dir
#     ):
#         last_checkpoint = get_last_checkpoint(training_args.output_dir)
#         if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
#             raise ValueError(
#                 f"Output directory ({training_args.output_dir}) already exists and is not empty. "
#                 "Use --overwrite_output_dir to overcome."
#             )
#         elif last_checkpoint is not None:
#             logger.info(
#                 f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
#                 "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
#             )

    # Tokenizer check: 해당 script는 Fast tokenizer를 필요로합니다.
    # if not isinstance(tokenizer, PreTrainedTokenizerFast):
    #     raise ValueError(
    #         "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
    #         "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
    #         "requirement"
    #     )

    # if data_args.max_seq_length > tokenizer.model_max_length:
    #     logger.warn(
    #         f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
    #         f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
    #     )
    # max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # if "validation" not in datasets:
    #     raise ValueError("--do_eval requires a validation dataset")
    # return last_checkpoint, max_seq_length


# loss funcion
# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.5, reduction="mean"):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction,
        )
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
    def __init__(self, classes=30, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


_criterion_entrypoints = {
    "CrossEntropy": nn.CrossEntropyLoss(),
    "focal": FocalLoss(),
    "label_smoothing": LabelSmoothingLoss(),
    "f1": F1Loss(),
}

def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]
