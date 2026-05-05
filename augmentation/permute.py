import random 
def permute_mslg(text,p=0.5):
    tokens=text.split()
    if len(tokens)< 4 or random.random()>p:
        return text
    verb_idx=1 if len(tokens)>1 else 0
    head=tokens[:verb_idx+1]
    tail=tokens[verb_idx+1:]
    random.shuffle(tail)
    return " ".join(head+tail)
FILLERS = {"A", "DE", "EL", "LA", "POR", "PARA", "QUE"}

def simplify_mslg(text, p=0.3):
    if random.random() > p:
        return text
    
    return " ".join([t for t in text.split() if t not in FILLERS])