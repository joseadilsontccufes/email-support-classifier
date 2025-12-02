import pandas as pd
import random
from faker import Faker
import numpy as np

fake = Faker("pt_BR")
random.seed(42)
np.random.seed(42)

categorias = {
    "Pagamentos e Faturamento": [
        "Fui cobrado duas vezes 😡",
        "Minha fatura veio errada",
        "Como atualizo meus dados de pagamento?",
        "Pagamento não foi reconhecido",
        "Preciso de reembolso urgente",
        "Cupom de desconto não funcionou",
        "Onde vejo meu recibo?"
    ],
    "Suporte Técnico": [
        "O sistema não está abrindo",
        "Erro ao tentar fazer login 😫",
        "Meu aplicativo trava toda hora",
        "Problema de conexão com o servidor",
        "Minha conta foi bloqueada sem motivo",
        "Site não carrega no navegador",
        "Sistema está muito lento hoje"
    ],
    "Trocas e Devoluções": [
        "Como faço pra devolver um produto?",
        "Recebi o item errado 😕",
        "Produto veio com defeito",
        "Troca ainda não foi aprovada",
        "Já enviei o produto e nada de resposta",
        "Quero trocar por outro tamanho",
        "Meu reembolso ainda não caiu"
    ],
    "Atendimento ao Cliente": [
        "Fui mal atendido no chat 😠",
        "Quero falar com o supervisor",
        "O atendente foi muito educado 👏",
        "Preciso atualizar meu endereço",
        "Não consegui resolver meu problema no chat",
        "Quero abrir uma reclamação formal",
        "Solicito retorno sobre meu atendimento"
    ],
    "Dúvidas Gerais": [
        "Vocês fazem entregas internacionais?",
        "Qual o prazo de entrega pra São Paulo?",
        "Onde posso consultar a política de privacidade?",
        "Vocês têm loja física?",
        "Tem desconto pra estudante?",
        "Qual o tempo de garantia dos produtos?",
        "Como entro em contato por telefone?"
    ],
    "Recursos Humanos": [
        "Como faço pra enviar meu currículo?",
        "Problemas no acesso ao portal do colaborador",
        "Não recebi meu holerite de setembro",
        "Como altero meus dados bancários?",
        "Erro ao registrar ponto no sistema",
        "Quero pedir trabalho remoto",
        "Documentos de admissão não aparecem"
    ],
    "Suporte de TI Interno": [
        "VPN não conecta de casa",
        "Impressora com erro de papel",
        "Computador não liga 😩",
        "Preciso reinstalar o sistema operacional",
        "Email corporativo não sincroniza",
        "Solicitação de acesso ao servidor",
        "Esqueci minha senha do Windows"
    ]
}

def adicionar_variacao(texto):
    texto = texto.replace("não", random.choice(["n", "nao", "nã"]))
    texto = texto.replace("pra", random.choice(["p/", "para", "pra"]))
    if random.random() < 0.25:
        texto = texto.replace("você", random.choice(["vc", "vcs", "cê"]))
    if random.random() < 0.15:
        texto = texto.replace("problema", random.choice(["bug", "erro", "issue"]))

    if random.random() < 0.25:
        texto += random.choice(["!", "!!", "...", "?!", " :)", " 😅", " 🤔", " 🙏", " 😔"])

    if random.random() < 0.2:
        texto = texto.lower()
    elif random.random() < 0.1:
        texto = texto.upper()

    return texto


def main():
    dados = []
    amostras_por_categoria = 10000 // len(categorias)

    for categoria, mensagens in categorias.items():
        for _ in range(amostras_por_categoria):
            assunto = adicionar_variacao(random.choice(mensagens))
            corpo = (
                f"{adicionar_variacao(fake.sentence(nb_words=random.randint(8,15)))} "
                f"{adicionar_variacao(random.choice(mensagens))}. "
                f"{adicionar_variacao(fake.paragraph(nb_sentences=random.randint(1,3)))}"
            )

            if random.random() < 0.05:
                corpo += " " + fake.sentence(nb_words=random.randint(4,8))

            dados.append({
                "subject": assunto,
                "body": corpo,
                "queue": categoria,
                "language": "pt"
            })

    df = pd.DataFrame(dados)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv("./data/emails_suporte_avancado.csv", index=False, encoding="utf-8")

    print(f"✅ Dataset criado com sucesso com {len(df)} e-mails realistas!")
    print(df.head())


if __name__ == "__main__":
    main()
