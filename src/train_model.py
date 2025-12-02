import pandas as pd
import numpy as np
import torch
import nlpaug.augmenter.word as naw

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)


def load_dataset():
    df = pd.read_csv("./data/emails_suporte_avancado.csv")
    df = df[df["language"].str.lower() == "pt"]
    df = df[["body", "queue"]].dropna()
    return df


def augment_data(df):
    aug = naw.SynonymAug(aug_src='wordnet', lang='por')

    augmented_texts = []
    augmented_labels = []

    for i, row in df.iterrows():
        original_text = row["body"]
        label = row["queue"]
        try:
            for _ in range(2):
                augmented = aug.augment(original_text)
                augmented_texts.append(augmented)
                augmented_labels.append(label)
        except Exception:
            continue

    df_aug = pd.DataFrame({"body": augmented_texts, "queue": augmented_labels})
    return pd.concat([df, df_aug], ignore_index=True)


def encode_data(df):
    texts = df["body"].astype(str).tolist()
    labels = df["queue"].astype(str).tolist()

    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)

    return texts, labels_enc, le


class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    logits, labels = eval_pred
    preds = logits.argmax(-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision_macro": precision_score(labels, preds, average="macro"),
        "recall_macro": recall_score(labels, preds, average="macro"),
    }


def main():
    df = load_dataset()
    df_full = augment_data(df)

    texts, labels, le = encode_data(df_full)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    model_name = "neuralmind/bert-base-portuguese-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    test_enc = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

    train_dataset = EmailDataset(train_enc, train_labels)
    test_dataset = EmailDataset(test_enc, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(le.classes_)
    )

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = trainer.evaluate()
    print("📊 Resultados:", results)

    preds = trainer.predict(test_dataset)
    pred_labels = preds.predictions.argmax(-1)

    print(
        classification_report(
            test_labels,
            pred_labels,
            target_names=le.classes_,
            labels=np.unique(test_labels),
        )
    )

    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")
    np.save("./model/label_encoder.npy", le.classes_)

    print("✅ Modelo salvo em ./model/")


if __name__ == "__main__":
    main()
