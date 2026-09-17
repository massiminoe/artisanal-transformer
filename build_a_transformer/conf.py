MAX_ABS_VAL = 100
CHARS = [str(i) for i in range(10)] + ["+", "-", "="]
VOCAB_SIZE = len(CHARS)
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}
EMBEDDING_DIM = 4  # Around sqrt(len(CHARS))