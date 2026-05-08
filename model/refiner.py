from transformers import BartForConditionalGeneration,BartTokenizer
class BARTRefiner:
    def __init__(self,model_name="facebook/bart-base",device="cuda"):
        self.tokenizer=BartTokenizer.from_pretrained(model_name)
        self.model=BartForConditionalGeneration.from_pretrained(model_name)
        self.device=device
        self.model.to(self.device)
    def refine(self,text,max_length=512):
        inputs=self.tokenizer(text,return_tensors="pt",truncation=True,padding=True)
        inputs={k:v.to(self.device) for k,v in inputs.items()}
        outputs=self.model.generate(
            **inputs,
            max_length=max_length,
        )
        return self.tokenizer.decode(outputs[0],skip_special_tokens=True)
