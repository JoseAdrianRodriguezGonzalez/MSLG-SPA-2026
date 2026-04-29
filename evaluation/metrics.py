import evaluate
from comet import download_model, load_from_checkpoint

comet_model = load_from_checkpoint(
    download_model("Unbabel/wmt22-comet-da")
)
bleu=evaluate.load("bleu")
meteor = evaluate.load("meteor")
chrf = evaluate.load("chrf")
import numpy as np
def build_compute_metrics(tokenizer,use_comet=True):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        print(type(preds))
        print(getattr(preds, "shape", "no shape"))
        print(preds[0][:10])
        if isinstance(preds, tuple):
            preds = preds[0]
        if len(preds.shape) == 3:
            # logits → argmax → token_ids
            preds = np.argmax(preds, axis=-1)
        
        preds=np.where(preds<0,tokenizer.pad_token_id,preds)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = [
            [(l if l != -100 else tokenizer.pad_token_id) for l in label]
            for label in labels
        ]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        bleu_score = bleu.compute(
            predictions=decoded_preds,
            references=[[l] for l in decoded_labels]
        )["bleu"]
        meteor_score=meteor.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )["meteor"]
        chrf_score = chrf.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )["score"]
        if use_comet:
            data = [
                {"src": "", "mt": p, "ref": r}
                for p, r in zip(decoded_preds, decoded_labels)
            ]
            comet_score = comet_model.predict(data, batch_size=8).system_score
        else:
            comet_score = 0.0
        accuracy = token_accuracy(decoded_preds, decoded_labels)
        return {
            "bleu": bleu_score,
            "meteor": meteor_score,
            "chrf": chrf_score,
            "commet":comet_score,
            "accuracy": accuracy
        }
    return compute_metrics
def token_accuracy(preds, refs):
    total = 0
    correct = 0
    for p, r in zip(preds, refs):
        p_tokens = p.split()
        r_tokens = r.split()
        for pt, rt in zip(p_tokens, r_tokens):
            if pt == rt:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0