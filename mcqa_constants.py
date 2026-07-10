import string

DATASET_PATH = "jchang153/copycolors_mcqa"
MODEL_NAME = "google/gemma-2-2b"
NUM_CHOICES = 4

TARGETS = ("answer_pointer", "answer_token")
FAMILY_ORDER = ("answer_pointer", "answer_token", "both")
ANSWER_LETTERS = tuple(string.ascii_uppercase)
