# FastAPI ML Project

Este é um projeto de estudo feito em **FastAPI**, que permite:
1. **Fazer upload** de datasets (arquivos `.csv`).
2. **Treinar modelos** de Machine Learning (ex.: RandomForest).
3. **Realizar predições** (incluindo probabilidades e batch inference).

> **Observação**: Este projeto tem fins **educacionais** e pode ser aprimorado em diversos aspectos. Sinta-se livre para contribuir!

---

## :rocket: Pré-requisitos

- **Python 3.8+** instalado na sua máquina.
- **pip** (gerenciador de pacotes) ou **conda** para instalar as dependências.
- [**Git**](https://git-scm.com/) (opcional, se quiser clonar o repositório diretamente).

---

## :package: Instalação

No diretório raiz do projeto, execute:

```bash
pip install -r requirements.txt
```

## Execução
Para subir a API localmente, rode:

```bash
uvicorn app.main:app --reload
```

A aplicação estará acessível em http://127.0.0.1:8000.
A documentação interativa do FastAPI estará em http://127.0.0.1:8000/docs ou http://127.0.0.1:8000/redoc.

## Endpoints Principais
1. Upload de Dataset
Endpoint: POST /datasets/upload
Body (form-data): arquivo .csv no campo file.
2. Treino de Modelo
Endpoint: POST /model/train
Body (JSON): Exemplo

```bash
{
  "file_name": "flowers.csv",
  "target_column": "species",
  "n_estimators": 100,
  "max_depth": null,
  "test_size": 0.2,
  "random_state": 42
}
```

Retorna model_path, report, accuracy, etc.

## Obtenção de Métricas
Endpoint: GET /metrics/
Query Param: file_name=flowers.csv

## Download do Modelo
Endpoint: GET /model/download-file
Query Param: model_name=model_flowers.joblib
Retorna um link para baixar o arquivo.

## Predições
Endpoint: POST /predict/
Body (JSON) para uma única amostra:

```bash
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Endpoint: POST /predict/batch
Body (JSON) para múltiplas amostras:

```bash
{
  "features_batch": [
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 3.4, 5.4, 2.3]
  ]
}
```

Endpoint: POST /predict/probabilities
Mesmo corpo do batch para retornar predict_proba

## Ideias de Melhoria
Validação Avançada de Dados

Verificar se o .csv contém a coluna-alvo informada ou se tem linhas vazias.
Adicionar logs de erro mais detalhados.
Suporte a Outros Modelos

Ex.: LogisticRegression, XGBoost, LightGBM, etc.
Parametrizar o tipo de modelo escolhido no /model/train.
Pipelines de Pré-processamento

Criar Pipeline com StandardScaler, OneHotEncoder etc. para usar em conjunto com o classificador.
Versionamento de Modelos

Salvar modelos com ID ou timestamp no nome, permitindo re-treinar sem sobrescrever o anterior.
Deploy em Produção

Usar Docker para empacotar o projeto, ou orquestrar em um serviço cloud (AWS, GCP, Azure).
Integração com Banco de Dados

Registrar histórico de treinamentos, métricas, logs, etc.