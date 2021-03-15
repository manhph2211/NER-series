import transformers


MAX_LEN = 128
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 3
BASE_MODEL_PATH = "./weights/bert_base_uncacsed"
MODEL_PATH = "./weights/model.pth"
TRAINING_FILE = "../data/ner_dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True
)