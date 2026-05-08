from tqdm import tqdm 
def generate_spa(dataset, model):
    synthetic = []

    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Generating SPA"):
        if row["task"] != "spa-mslg":
            continue

        spa = row["input_text"]

        mslg_pred = model.spa_to_mslg(spa)
        spa_recon = model.mslg_to_spa(mslg_pred)

        synthetic.append({
            "input_text": spa,
            "target_text": mslg_pred,
            "task": "spa-mslg",
            "spa_recon": spa_recon,
            "source": "synthetic"
        })

    return synthetic
def generate_mslg(dataset, model):
    synthetic = []

    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Generating MSLG"):
        if row["task"] != "mslg-spa":
            continue

        mslg = row["input_text"]

        spa_pred = model.mslg_to_spa(mslg)
        mslg_recon = model.spa_to_mslg(spa_pred)

        synthetic.append({
            "input_text": mslg,
            "target_text": spa_pred,
            "task": "mslg-spa",
            "mslg_recon": mslg_recon,
            "source": "synthetic"
        })

    return synthetic