import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./model")
    model = AutoModelForSequenceClassification.from_pretrained("./model")
    classes = np.load("./model/label_encoder.npy", allow_pickle=True)
    return tokenizer, model, classes


def predict_queue(text):
    tokenizer, model, classes = load_model()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_id = torch.argmax(outputs.logits, dim=1).item()
    return classes[predicted_id]


if __name__ == "__main__":
    email = "Prezados, estou com problemas para acessar minha conta de e-mail corporativo."
    print(predict_queue(email))
