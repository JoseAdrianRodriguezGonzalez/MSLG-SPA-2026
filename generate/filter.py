from difflib import SequenceMatcher

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def filter_synthetic(data, threshold=0.8):
    filtered = []

    for row in data:
        if row["task"] == "spa-mslg":
            sim = similarity(row["input_text"], row["spa_recon"])

            if sim >= threshold:
                filtered.append({
                    "SPA": row["input_text"],
                    "MSLG": row["target_text"]
                })

        else:  # mslg-spa
            sim = similarity(row["input_text"], row["mslg_recon"])

            if sim >= threshold:
                filtered.append({
                    "SPA": row["target_text"],
                    "MSLG": row["input_text"]
                })

    return filtered