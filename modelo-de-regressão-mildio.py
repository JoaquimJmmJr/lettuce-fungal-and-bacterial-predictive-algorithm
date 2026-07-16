import os
import numpy as np
import pandas as pd

# ================= Fonte de dados ================= #
# "csv"   -> leitura local para testes (dados_sensores.csv)
# "mysql" -> leitura do banco MySQL (objetivo final, quando o pipeline ESP32 → MySQL estiver ativo)
FONTE_DADOS = os.getenv("FONTE_DADOS", "csv")

CSV_PATH = "dados_sensores.csv"

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "lettuce_mildio"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
}
MYSQL_TABLE = os.getenv("MYSQL_TABLE", "leituras_sensores")


def carregar_dados_csv(caminho=CSV_PATH):
    df = pd.read_csv(caminho, sep=";")
    df = df.loc[:, ~df.columns.str.fullmatch(r"Unnamed.*")]
    df.columns = df.columns.str.strip()
    return df


def carregar_dados_mysql(config=MYSQL_CONFIG, tabela=MYSQL_TABLE):
    import mysql.connector

    conexao = mysql.connector.connect(**config)
    try:
        query = f"SELECT timestamp, temp, umidade FROM {tabela}"
        df = pd.read_sql(query, conexao)
    finally:
        conexao.close()
    return df


def carregar_dados(fonte=FONTE_DADOS):
    if fonte == "csv":
        return carregar_dados_csv()
    elif fonte == "mysql":
        return carregar_dados_mysql()
    else:
        raise ValueError(f"Fonte de dados desconhecida: {fonte!r} (use 'csv' ou 'mysql')")


data = carregar_dados()

# Regras agronômicas para definir o risco de ocorrência do Míldio da alface (Downey mildew on lettuce)
## Risco 0 (baixo risco): caso contrário
## Risco 1 (médio risco): umidade > 85%
## Risco 2 (alto risco): umidade > 90% e temperatura entre 10 e 20°C

def calcular_risco(row):
    if row["umidade"] > 90 and 10 <= row["temp"] <= 20:
        return 2
    elif row["umidade"] > 85:
        return 1
    else:
        return 0

# Crie uma nova coluna "risco" aplicando a função calcular_risco a cada linha do DataFrame:
data["risco"] = data.apply(calcular_risco, axis=1)


# ================= Treinamento do modelo de regressão logística ================= #
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = data[["temp", "umidade"]]
y = data["risco"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Acurácia:", model.score(X_test, y_test))

# ================= Teste da previsão com cenários reais ================= #
novo = pd.DataFrame({
    "temp": [18],
    "umidade": [93]
})

print(model.predict(novo))
