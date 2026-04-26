from transformers import DataCollatorForSeq2Seq
def get_collator(tokenizer,model):
    return DataCollatorForSeq2Seq(tokenizer=tokenizer,model=model)