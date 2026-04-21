import re
#Funciones para normalziar el MSLG 
def normalize_mslg(text):
    text=text.upper()
    return text 
def clean_prefixes(text):
    return re.sub(r'\b(?:dm|pro|xx|etc)-([A-Za-zÁÉÍÓÚÑáéíóúñ]+)', r'\1', text)
def normalize_spaces(text):
    return " ".join(text.split())
def pipeline_mslg(text):
    text=clean_prefixes(text)
    text=normalize_mslg(text)
    text=normalize_spaces(text)
    return text