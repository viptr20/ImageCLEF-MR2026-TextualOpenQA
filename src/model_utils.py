from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer

def load_model(model_name: str):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model