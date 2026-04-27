def clasify_error(pred,ref):
    if pred==ref:
        return "correct"
    if len(pred.split())<len(ref.split()):
        return "tokens faltantes"
    if len(pred.split())>len(ref.split()):
        return "extra tokens"
    if pred.split()!=ref.split():
        return "orden"
    return "ambiguedad"