import pandas as pd
from augmentation.fasttext_augmenter import FastTextAugmenter
from augmentation.permute import permute_mslg, simplify_mslg

def augment_df(df, ft_model_path=None):
    augmenter = None
    if ft_model_path:
        augmenter = FastTextAugmenter(ft_model_path)
    rows = []
    for _, row in df.iterrows():
        src = row["input_text"]
        tgt = row["target_text"]
        # ORIGINAL
        rows.append(row)
        # PERMUTACIÓN
        rows.append({
            **row,
            "input_text": permute_mslg(src)
        })
        # SIMPLIFICACIÓN
        rows.append({
            **row,
            "input_text": simplify_mslg(src)
        })
        # LÉXICO (fastText)
        if augmenter:
            rows.append({
                **row,
                "input_text": augmenter.augment_text(src)
            })
    df_aug = pd.DataFrame(rows)
    return df_aug.sample(frac=1).reset_index(drop=True)  # shuffle final