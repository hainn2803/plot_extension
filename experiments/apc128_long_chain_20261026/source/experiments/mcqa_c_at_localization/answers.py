"""The single canonical answer continuation after the frozen 'Answer:' prompt."""
from .protocol import LABELS


def answer_ids(tokenizer):
    encoded = [tokenizer.encode(" " + label, add_special_tokens=False) for label in LABELS]
    if any(len(tokens) != 1 for tokens in encoded):
        raise ValueError("Each space-prefixed answer must be one token")
    return dict(zip(LABELS, (tokens[0] for tokens in encoded)))


def corrected_token_map(original, tokenizer):
    result = dict(original)
    bare = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
    expected = [tokens[0] for tokens in bare]
    corrected = list(answer_ids(tokenizer).values())
    if original["label_ids"] not in (expected, corrected):
        raise ValueError("Parent answer-token convention is unrecognized")
    prefix = tokenizer.encode(original["reference"]["text"], add_special_tokens=True)
    for label, token in answer_ids(tokenizer).items():
        continued = tokenizer.encode(original["reference"]["text"] + " " + label, add_special_tokens=True)
        if continued != prefix + [token]:
            raise ValueError("Answer continuation changes the frozen prompt tokenization")
    result.update(label_ids=corrected, answer_continuation_prefix=" ")
    return result
