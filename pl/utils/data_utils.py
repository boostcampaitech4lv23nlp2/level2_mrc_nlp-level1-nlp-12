import collections

from typing import Any, Tuple

import numpy as np
import torch
import transformers
import os
import json
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
)
from tqdm.auto import tqdm
from transformers import EvalPrediction
from retrievals.base_retrieval import SparseRetrieval
def prepare_train_features(examples, tokenizer):
    # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
    # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_token_type_ids=False,  # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
        padding="max_length",
    )

    # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # token의 캐릭터 단위 position를 찾을 수 있도록 offset mapping을 사용합니다.
    # start_positions과 end_positions을 찾는데 도움을 줄 수 있습니다.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # 데이터셋에 "start position", "enc position" label을 부여합니다.
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)  # cls index

        # sequence id를 설정합니다 (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # 하나의 example이 여러개의 span을 가질 수 있습니다.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        # answer가 없을 경우 cls_index를 answer로 설정합니다(== example에서 정답이 없는 경우 존재할 수 있음).
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # text에서 정답의 Start/end character index
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # text에서 current span의 Start token index
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # text에서 current span의 End token index
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # 정답이 span을 벗어났는지 확인합니다(정답이 없는 경우 CLS index로 label되어있음).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # token_start_index 및 token_end_index를 answer의 끝으로 이동합니다.
                # Note: answer가 마지막 단어인 경우 last offset을 따라갈 수 있습니다(edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_features(examples, tokenizer):
    # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
    # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_token_type_ids=False,  # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
        padding="max_length",
    )

    # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
    # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # sequence id를 설정합니다 (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # 하나의 example이 여러개의 span을 가질 수 있습니다.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping을 None으로 설정해서 token position이 context의 일부인지 쉽게 판별 할 수 있습니다.
        # tokenized_examples["offset_mapping"][i] = [
        #     (o if sequence_ids[k] == context_index else 0)
        #     for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        # ]
    return tokenized_examples


def postprocess_qa_predictions(
    examples,
    features,
    id,
    predictions: Tuple[np.ndarray, np.ndarray],
    output_dir = None,
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    mode = 'eval',
    null_score_diff_threshold: float = 0.0,
):
    """
    Post-processes : qa model의 prediction 값을 후처리하는 함수
    모델은 start logit과 end logit을 반환하기 때문에, 이를 기반으로 original text로 변경하는 후처리가 필요함

    Args:
        examples: 전처리 되지 않은 데이터셋 (see the main script for more information).
        features: 전처리가 진행된 데이터셋 (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            모델의 예측값 :start logits과 the end logits을 나타내는 two arrays              첫번째 차원은 :obj:`features`의 element와 갯수가 맞아야함.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            정답이 없는 데이터셋이 포함되어있는지 여부를 나타냄
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            답변을 찾을 때 생성할 n-best prediction 총 개수
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            생성할 수 있는 답변의 최대 길이
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            null 답변을 선택하는 데 사용되는 threshold
            : if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).
            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            아래의 값이 저장되는 경로
            dictionary : predictions, n_best predictions (with their scores and logits) if:obj:`version_2_with_negative=True`,
            dictionary : the scores differences between best and null answers
        prefix (:obj:`str`, `optional`):
            dictionary에 `prefix`가 포함되어 저장됨
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            이 프로세스가 main process인지 여부(logging/save를 수행해야 하는지 여부를 결정하는 데 사용됨)
    """
    assert len(predictions) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    # assert len(predictions[0]) == len(
    #     features
    # ), print(f"Got {len(predictions[0])} predictions and {len(features)} features.")
    # print(predictions[0], features)
    # example과 mapping되는 feature 생성
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(id):
        features_per_example[example_id_to_index[feature]].append(i)

    # prediction, nbest에 해당하는 OrderedDict 생성합니다.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # 전체 example들에 대한 main Loop
    for example_index, example in enumerate(tqdm(examples)):
        # 해당하는 현재 example index
        feature_indices = features_per_example[example_index]
        min_null_prediction = 0
        prelim_predictions = []

        # 현재 example에 대한 모든 feature 생성합니다.
        for feature_index in feature_indices:
            # 각 featureure에 대한 모든 prediction을 가져옵니다.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # logit과 original context의 logit을 mapping합니다.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional : `token_is_max_context`, 제공되는 경우 현재 기능에서 사용할 수 있는 max context가 없는 answer를 제거합니다
            token_is_max_context = features[feature_index].get(
                "token_is_max_context", 0
            )

            # minimum null prediction을 업데이트 합니다.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction == 0 or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # `n_best_size`보다 큰 start and end logits을 살펴봅니다.
            start_indexes = torch.argsort(start_logits, descending=True)[0 : n_best_size + 1].tolist()

            end_indexes = torch.argsort(end_logits, descending=True)[0 : n_best_size + 1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # out-of-scope answers는 고려하지 않습니다.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or (offset_mapping[start_index] == 0 and offset_mapping[end_index] == 0)
                    ):
                        continue
                    # 길이가 < 0 또는 > max_answer_length인 answer도 고려하지 않습니다.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # 최대 context가 없는 answer도 고려하지 않습니다.
                    if (
                        token_is_max_context != 0
                        and not token_is_max_context.get(str(start_index), False)
                    ):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        if version_2_with_negative:
            # minimum null prediction을 추가합니다.
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # 가장 좋은 `n_best_size` predictions만 유지합니다.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # 낮은 점수로 인해 제거된 경우 minimum null prediction을 다시 추가합니다.
        if version_2_with_negative and not any(p["offsets"] == (0, 0) for p in predictions):
            predictions.append(min_null_prediction)

        # offset을 사용하여 original context에서 answer text를 수집합니다.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # rare edge case에는 null이 아닌 예측이 하나도 없으며 failure를 피하기 위해 fake prediction을 만듭니다.
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):

            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # 모든 점수의 소프트맥스를 계산합니다(we do it with numpy to stay independent from torch/tf in this file, using the LogSumExp trick).
        # scores = np.array([pred.pop("score") for pred in predictions])
        # exp_scores = np.exp(scores - np.max(scores))
        # probs = exp_scores / exp_scores.sum()

        # # 예측값에 확률을 포함합니다.
        # for prob, pred in zip(probs, predictions):
        #     pred["probability"] = prob

        # best prediction을 선택합니다.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # else case : 먼저 비어 있지 않은 최상의 예측을 찾아야 합니다
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # threshold를 사용해서 null prediction을 비교합니다.
            score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[example["id"]] = float(score_diff)  # JSON-serializable 가능
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # np.float를 다시 float로 casting -> `predictions`은 JSON-serializable 가능
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]
    if mode == 'predict':
        output_dir = '/opt/ml/input/code/predictions'
        prediction_file = os.path.join(
            output_dir,
            "predictions.json",
        )
        with open(prediction_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n"
            )
    return all_predictions


def post_processing_function(id, predictions, tokenizer, mode, path):
    # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
    if mode == "eval":
        examples = load_from_disk(path)
        examples = examples['validation']
        features = examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=4,
            remove_columns=examples.column_names,
            fn_kwargs={"tokenizer": tokenizer},
        )
    else:
        examples = load_from_disk(path)
        examples = run_sparse_retrieval(tokenizer.tokenize, examples, 'predict', False)
        column_names = examples['validation'].column_names
        examples = examples['validation']
        features = examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=4,
            remove_columns=column_names,
            fn_kwargs={"tokenizer": tokenizer},
        )
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        id=id,
        predictions=predictions,
        max_answer_length=100,
        mode=mode,
    )
    # Metric을 구할 수 있도록 Format을 맞춰줍니다.
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    if mode == "predict":
        return formatted_predictions

    elif mode == "eval":
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

def run_sparse_retrieval(
    tokenize_fn,
    datasets,
    mode,
    use_faiss,
    data_path: str = "../../data",
    context_path: str = "wikipedia_documents.json",
):

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval(tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path)
    retriever.get_sparse_embedding()

    if use_faiss:
        retriever.build_faiss(num_clusters=64)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=10
        )
    else:
        df = retriever.retrieve(datasets["validation"], topk=10)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if mode=='predict':
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif mode == 'eval':
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets