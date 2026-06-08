from typing import List, Dict
import evaluate

meteor_metric = evaluate.load("meteor")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    import numpy as np
    from transformers import PreTrainedTokenizerBase

    preds = predictions
    labels = labels

    tokenizer: PreTrainedTokenizerBase = compute_metrics.tokenizer

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    meteor = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"meteor": meteor["meteor"]}

# will be set from train.py
compute_metrics.tokenizer = None