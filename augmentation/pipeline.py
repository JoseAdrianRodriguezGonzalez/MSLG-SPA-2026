import pandas as pd
from augmentation.fasttext_augmenter import FastTextAugmenter
from augmentation.permute import permute_mslg, simplify_mslg

def augment_df(df, ft_model_path=None):
    augmenter = FastTextAugmenter(ft_model_path) if ft_model_path else None
    
    rows = []

    for _, row in df.iterrows():
        src = row["input_text"]
        task = row["task"]
        rows.append(row)
        if task == "mslg-spa":
            rows.append({**row, "input_text": permute_mslg(src)})
            rows.append({**row, "input_text": simplify_mslg(src)})

        elif task == "spa-mslg":
        
            if augmenter:
                rows.append({
                    **row,
                    "input_text": augmenter.augment_text(src)
                })

    return pd.DataFrame(rows).sample(frac=1).reset_index(drop=True)