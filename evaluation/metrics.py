import evaluate
bleu=evaluate.load("bleu")
def build_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = [
            [(l if l != -100 else tokenizer.pad_token_id) for l in label]
            for label in labels
        ]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = bleu.compute(
            predictions=decoded_preds,
            references=[[l] for l in decoded_labels]
        )
        accuracy = structural_accuracy(decoded_preds, decoded_labels)
        return {
            "bleu": result["bleu"],
            "accuracy": accuracy
        }
    return compute_metrics
def structural_accuracy(preds, refs):
    correct = 0
    for p, r in zip(preds, refs):
        if p.strip() == r.strip():
            correct += 1
    return correct / len(preds)