import re
import unicodedata

def strip_accents(text):
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )

def normalize_mslg_aggresive(text: str) -> str:
    text = text.upper().strip()
    text = re.sub(r"\s+", " ", text)
    tokens = text.split()
    tokens_no_acc = [strip_accents(t) for t in tokens]
    for i in range(len(tokens_no_acc) - 1):
        if tokens_no_acc[i] == "POR" and tokens_no_acc[i+1] == "QUE":
            tokens.pop(i)
            tokens.pop(i)
            tokens = ["POR", "QUÉ"] + tokens
            break
    return " ".join(tokens)
def normalize_spa(text: str) -> str:
    text = text.strip()
    text = text.replace("¿", " ¿ ").replace("?", " ? ")
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.,!]", "", text)
    return text.strip()
def normalize_by_task(row):
    task = row["task"]
    if task == "mslg-spa":
        row["input_text"] = normalize_mslg_aggresive(row["input_text"])
        row["target_text"] = normalize_spa(row["target_text"])

    elif task == "spa-mslg":
        row["input_text"] = normalize_spa(row["input_text"])
        row["target_text"] = normalize_mslg_aggresive(row["target_text"])

    return row