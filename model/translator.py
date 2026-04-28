from transformers import T5Tokenizer,T5ForConditionalGeneration
import torch
class MSLGTranslator:
    def __init__(self,model_name="t5-small"):
        self.tokenizer=T5Tokenizer.from_pretrained(model_name)
        self.model=T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def preprocess(self,text):
        return text.strip()
    def _generate(self,text,max_len=64,beams=4):
        input=self.tokenizer(text,return_tensors="pt",truncation=True).to(self.device)
        output=self.model.generate(
            **input,
            max_length=max_len,
            num_beams=beams
        )
        return self.tokenizer.decode(output[0],skip_special_tokens=True)
    #mslgspa
    def mslg_to_spa(self,text):
        text=self.preprocess(text)
        prompt=f"translate MSLG to Spanish: {text}"
        return self._generate(prompt)
    #mspa mslg
    def spa_to_mslg(self,text):
        text=self.preprocess(text)
        prompt=f"translate spanish to mslg: {text}"
        return self._generate(prompt)