# 📧 Email Support Classifier

Classificador automático de e-mails de suporte utilizando Machine Learning.
Este projeto permite gerar datasets rotulados, treinar um modelo e realizar previsões de categoria para novos e-mails.

## 🚀 Visão Geral

O objetivo deste projeto é automatizar a classificação de e-mails de suporte técnico, agrupando mensagens por categorias para agilizar o atendimento.

O pipeline inclui:

1. Geração de dataset a partir de diretórios de e-mails.
2. Treinamento de modelo de ML para classificação.
3. Predição de categorias para novos e-mails.

## 📁 Estrutura do Projeto

```
email-support-classifier/
├── src/
│   ├── generate_dataset.py
│   ├── train_model.py
│   ├── predict.py
├── requirements.txt
└── README.md
```

## 🔧 Instalação

```bash
git clone https://github.com/joseadilsontccufes/email-support-classifier.git
cd email-support-classifier
pip install -r requirements.txt
```

## 🧱 Estrutura dos Dados

```
emails/
├── categoria1/
│   ├── email1.txt
│   ├── email2.txt
├── categoria2/
│   ├── email1.txt
│   ├── email2.txt
```

## 🧰 Como Utilizar

### 1. Gerar Dataset

```bash
python src/generate_dataset.py --input_dir ./emails --output_file dataset.csv
```

### 2. Treinar Modelo

```bash
python src/train_model.py --dataset dataset.csv --model_file model.pkl
```

### 3. Fazer Previsões

```bash
python src/predict.py --model_file model.pkl --email "mensagem aqui"
```

## 🧠 Sobre o Modelo

- Vetorização: TF-IDF
- Classificador: LinearSVC
- Métricas exibidas no terminal

## 📊 Formato do Dataset

| texto_do_email | categoria |
| -------------- | --------- |

## 🤝 Contribuição

1. Fork
2. Branch
3. Commit
4. Pull Request

## 📄 Licença

MIT

## 👤 Autor

José Adilson
GitHub: https://github.com/joseadilsontccufes
