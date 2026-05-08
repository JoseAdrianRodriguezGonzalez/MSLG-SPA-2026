from transformers import BartForConditionalGeneration,BartTokenizer
class BARTRefiner:
    def __init__(self,model_name="facebook/bart-base",device="cuda"):
        self.tokenizer