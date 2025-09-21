_TRAIN = {"train","training","trn"}
_VAL   = {"val","valid","validation","eval","evaluation","dev"}
_TEST  = {"test","testing","tst"}

def normalize_split(name: str) -> str:
    n = (name or "").lower()
    if n in _TRAIN or any(n.startswith(x) for x in _TRAIN): return "train"
    if n in _VAL   or any(n.startswith(x) for x in _VAL):   return "val"
    if n in _TEST  or any(n.startswith(x) for x in _TEST):  return "test"
    return "train"
