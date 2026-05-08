from model.refiner import BARTRefiner
from tqdm import tqdm

def refine_with_bart(pairs):
    refiner = BARTRefiner()
    refined_data = []
    for sample in tqdm(pairs):
        src = sample["input_text"]
        tgt = sample["target_text"]
        refined_tgt = refiner.refine(tgt)
        new_sample = sample.copy()
        new_sample["target_text"] = refined_tgt
        new_sample["refined"] = True
        refined_data.append(new_sample)
    return refined_data