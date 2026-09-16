# Medical Triage NLP


[![CI/CD Medical Triage](https://github.com/Gabrieln91Costa/tech-challenge-medical-triage/actions/workflows/ci.yml/badge.svg)](https://github.com/Gabrieln91Costa/tech-challenge-medical-triage/actions/workflows/ci.yml)


PDF -  (Pasta docs) - Medical_Triage_NLP_Cenario_Arquitetura_e_Requisitos.pdf


Projeto de **Machine Learning aplicado à classificação de textos médicos**, desenvolvido para o **Tech Challenge – Deploy de Modelo em Produção com Pipeline CI/CD, Monitoramento e Otimização de Latência**.

> **Observação importante:** o dataset utilizado possui 5 categorias de condições médicas e não possui rótulos de urgência clínica (`normal`, `atenção`, `urgente`). Portanto, o projeto demonstra tecnicamente o pipeline de classificação de textos médicos e os requisitos de MLOps, sem afirmar capacidade de triagem clínica real.

---

# 1. Objetivo

O projeto tem como objetivo construir um pipeline completo para disponibilização de um modelo de Machine Learning como serviço, contemplando:

- Classificação automática de textos médicos;
- Treinamento e avaliação do modelo;
- Disponibilização da inferência através de API REST;
- Containerização com Docker;
- Pipeline CI/CD com GitHub Actions;
- Treinamento/retreinamento através do Apache Airflow;
- Monitoramento com Prometheus;
- Dashboard de observabilidade com Grafana;
- Otimização da inferência utilizando ONNX Runtime;
- Medição comparativa de latência e throughput.

O projeto foi estruturado para demonstrar o ciclo de vida do modelo desde o treinamento até sua disponibilização e monitoramento.

## Principais funcionalidades

- Classificação de textos médicos;
- API REST utilizando FastAPI;
- Endpoint de saúde `/health`;
- Endpoint de predição `/predict`;
- Endpoint de métricas `/metrics`;
- Treinamento utilizando Scikit-Learn;
- Conversão do modelo para ONNX;
- Inferência otimizada com ONNX Runtime validada separadamente;
- Testes automatizados com Pytest;
- Lint automatizado com Ruff;
- Build da imagem Docker;
- DAG de treinamento/re-treinamento com Airflow;
- Coleta de métricas com Prometheus;
- Visualização das métricas com Grafana;
- Benchmark de latência;
- Benchmark de throughput.

---

# 2. Arquitetura

A solução utiliza uma arquitetura orientada a serviços, adequada ao objetivo do Tech Challenge.

A arquitetura foi dividida em componentes com responsabilidades específicas:

```text
                         ┌─────────────────────┐
                         │      GitHub         │
                         │   Actions CI/CD     │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    │               │                │
                  Lint            Tests          Docker Build
                    │               │                │
                    └───────────────┴────────────────┘

┌────────────────┐
│   Dataset CSV  │
└───────┬────────┘
        │
        ▼
┌────────────────────┐
│      Airflow       │
│ Training / Retrain │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Scikit-Learn Model │
│ TF-IDF + Logistic  │
│ Regression         │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│    ONNX Runtime    │
│ Latência otimizada │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│      FastAPI       │
│      :8000         │
└───────┬─────┬──────┘
        │     │
        │     └──────────────► /metrics
        │                         │
        ▼                         ▼
   /predict                 ┌──────────────┐
                             │ Prometheus   │
                             │    :9090     │
                             └──────┬───────┘
                                    │
                                    ▼
                             ┌──────────────┐
                             │   Grafana    │
                             │    :3000     │
                             └──────────────┘
```

## Tecnologias utilizadas

- Python 3.11;
- FastAPI;
- Scikit-Learn;
- Pandas;
- NumPy;
- Joblib;
- ONNX;
- ONNX Runtime;
- skl2onnx;
- Pytest;
- Ruff;
- Docker;
- Docker Compose;
- Apache Airflow;
- Prometheus;
- Grafana;
- GitHub Actions.
- Azure.

---

# 3. Estrutura do projeto

A estrutura foi organizada separando API, treinamento, testes, monitoramento e automação:

```text
tech-challenge-medical-triage/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── model.py
│
├── data/
│   ├── medical_tc_train.csv
│   ├── medical_tc_test.csv
│   └── medical_tc_labels.csv
│
├── model/
│   ├── model.pkl
│   └── model.onnx
│
├── training/
│   ├── train.py
│   ├── convert_to_onnx.py
│   ├── test_onnx.py
│   ├── benchmark_latency.py
│   └── benchmark_throughput.py
│
├── tests/
│   └── test_api.py
│
├── monitoring/
│   └── prometheus/
│       └── prometheus.yml
│
├── airflow/
│   ├── Dockerfile
│   └── dags/
│       └── medical_triage_training.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .gitignore
├── README.md
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Responsabilidade dos diretórios

### `app/`

Contém a aplicação FastAPI e a camada responsável pela inferência do modelo.

### `training/`

Concentra o ciclo de Machine Learning:

- treinamento;
- conversão para ONNX;
- validação da previsão;
- benchmark de latência;
- benchmark de throughput.

### `data/`

Contém os datasets utilizados no treinamento e avaliação.

### `model/`

Armazena os artefatos gerados pelo treinamento.

### `tests/`

Contém os testes automatizados da API.

### `monitoring/`

Contém a configuração do Prometheus.

### `airflow/`

Contém o ambiente e a DAG responsável pelo treinamento/re-treinamento.

### `.github/workflows/`

Contém o pipeline de CI/CD executado pelo GitHub Actions.

---

# 4. Modelo de Machine Learning

## 4.1 Modelo escolhido

Foi utilizado um pipeline de classificação composto por:

```text
Texto médico
     │
     ▼
TF-IDF
     │
     ▼
Logistic Regression
     │
     ▼
Classe prevista
```

Implementação:

```python
Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42,
    )),
])
```

## 4.2 Justificativa da escolha

O **TF-IDF** foi escolhido para transformar os textos médicos em uma representação numérica baseada na importância dos termos.

O uso de `ngram_range=(1, 2)` permite considerar:

- palavras individuais;
- combinações de duas palavras.

O `max_features=10000` limita o vocabulário e ajuda a controlar o custo computacional.

A **Logistic Regression** foi utilizada como classificador por apresentar uma abordagem simples, leve e adequada como baseline para classificação de textos.

Essa combinação é adequada ao objetivo do Tech Challenge porque permite:

- treinamento rápido;
- inferência rápida;
- baixo consumo de recursos;
- fácil disponibilização como API;
- conversão para ONNX;
- comparação objetiva de performance.

---

# 5. Dataset

Foi utilizado o **Medical Abstracts TC Corpus**.

Arquivos:

```text
data/
├── medical_tc_train.csv
├── medical_tc_test.csv
└── medical_tc_labels.csv
```

## Dataset de treinamento

```text
11.550 registros
2 colunas
```

Colunas:

```text
condition_label
medical_abstract
```

## Dataset de teste

```text
2.888 registros
```

## Classes

O dataset contém 5 categorias:

| Código | Categoria |
|---:|---|
| 1 | neoplasms |
| 2 | digestive system diseases |
| 3 | nervous system diseases |
| 4 | cardiovascular diseases |
| 5 | general pathological conditions |

> Essas categorias representam condições médicas e não níveis de urgência. Para uma aplicação clínica real de triagem seria necessário um dataset com rótulos de urgência definidos por critérios clínicos apropriados.

---

# 6. Treinamento

O treinamento é realizado através do arquivo:

```text
training/train.py
```

Execução:

```bash
python training/train.py
```

O script:

1. Carrega o dataset de treinamento;
2. Separa texto e variável alvo;
3. Cria o pipeline TF-IDF + Logistic Regression;
4. Treina o modelo;
5. Avalia utilizando o dataset de teste;
6. Exibe as métricas;
7. Salva o modelo em:

```text
model/model.pkl
```

## Resultado obtido

A avaliação realizada apresentou:

```text
Accuracy: 0.5471
```

O resultado demonstra a capacidade do baseline de classificação sobre o dataset utilizado, mas não deve ser interpretado como métrica de segurança ou desempenho clínico.

---

# 7. API REST

A API foi desenvolvida utilizando **FastAPI**.

## Executar localmente

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

A API ficará disponível em:

```text
http://localhost:8000
```

## Swagger

```text
http://localhost:8000/docs
```

## ReDoc

```text
http://localhost:8000/redoc
```

---

# 8. Endpoints

## Health Check

```http
GET /health
```

Resposta:

```json
{
  "status": "ok"
}
```

## Predição

```http
POST /predict
```

Request:

```json
{
  "text": "Patient with cardiovascular disease and severe chest pain"
}
```

Resposta:

```json
{
  "prediction": 4,
  "class_name": "cardiovascular diseases",
  "latency_ms": 7.5858
}
```

O campo `latency_ms` representa o tempo medido na execução da predição.

## Métricas

```http
GET /metrics
```

O endpoint disponibiliza as métricas no formato utilizado pelo Prometheus.

---

# 9. Monitoramento

A aplicação utiliza:

```text
FastAPI
   │
   ▼
prometheus-client
   │
   ▼
/metrics
   │
   ▼
Prometheus
   │
   ▼
Grafana
```

As métricas instrumentadas incluem:

- quantidade de requisições;
- status HTTP;
- latência das requisições.

## Prometheus

O Prometheus realiza o scrape da API a cada:

```text
5 segundos
```

Configuração:

```text
monitoring/prometheus/prometheus.yml
```

O target utilizado dentro do Docker Compose é:

```text
api:8000
```

## Grafana

O Grafana utiliza o Prometheus como fonte de dados.

Dentro do ambiente Docker, a URL do datasource deve ser:

```text
http://prometheus:9090
```

e não:

```text
http://localhost:9090
```

pois `localhost` dentro do container do Grafana representa o próprio container.

---

# 10. Dashboard Grafana

O dashboard deve contemplar pelo menos três indicadores principais.

## 10.1 Total de requisições

```promql
sum(http_requests_total{endpoint="/predict"})
```

## 10.2 Latência média

```promql
1000 *
sum(rate(http_request_duration_seconds_sum{endpoint="/predict"}[1m]))
/
sum(rate(http_request_duration_seconds_count{endpoint="/predict"}[1m]))
```

Resultado apresentado em milissegundos.

## 10.3 Taxa de erro

```promql
100 *
sum(rate(http_requests_total{endpoint="/predict",status=~"5.."}[1m]))
/
sum(rate(http_requests_total{endpoint="/predict"}[1m]))
```

---

# 11. CI/CD – GitHub Actions

O projeto possui pipeline automatizado através de:

```text
.github/workflows/ci.yml
```

O fluxo implementado é:

```text
Push / Pull Request
        │
        ▼
      Lint
        │
        ▼
  Train Model + Tests
        │
        ▼
   Docker Build
```

## Etapas

### Lint

Utiliza:

```text
Ruff
```

Execução:

```bash
ruff check .
```

### Testes

Utiliza:

```text
Pytest
```

Execução:

```bash
pytest -v
```

### Docker Build

A pipeline realiza o build da imagem:

```bash
docker build -t medical-triage-api:${{ github.sha }} .
```

O modelo é treinado antes do build para garantir que o artefato necessário esteja disponível no processo de construção.

---

# 12. Testes

Os testes estão localizados em:

```text
tests/test_api.py
```

Atualmente são contemplados:

- health check;
- endpoint de predição;
- resposta para texto vazio;
- endpoint de métricas.

Executar:

```bash
pytest -v
```

Resultado local validado:

```text
4 passed
```

---

# 13. Airflow – Treinamento e Retreinamento

O Apache Airflow foi utilizado para representar o processo de treinamento/re-treinamento do modelo.

DAG:

```text
airflow/dags/medical_triage_training.py
```

Fluxo:

```text
read_dataset
      │
      ▼
train_model
      │
      ▼
model/model.pkl
```

## Task 1 – Leitura do dataset

A task:

- carrega o CSV;
- valida se o dataset está vazio;
- verifica as colunas obrigatórias;
- registra informações no log.

## Task 2 – Treinamento

A task:

- carrega os dados;
- treina o pipeline;
- salva o modelo em:

```text
/opt/airflow/project/model/model.pkl
```

A DAG foi executada com sucesso no ambiente local.

---

# 14. Docker

O projeto possui um `Dockerfile` para a API.

Build:

```bash
docker build -t medical-triage-api .
```

Execução:

```bash
docker run -p 8000:8000 medical-triage-api
```

A API ficará disponível em:

```text
http://localhost:8000
```

---

# 15. Docker Compose

O Docker Compose orquestra os serviços do projeto:

```text
docker-compose.yml
```

Serviços:

```text
┌──────────────────────────┐
│          API             │
│        :8000             │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│       Prometheus         │
│         :9090            │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│         Grafana          │
│         :3000            │
└──────────────────────────┘

┌──────────────────────────┐
│         Airflow          │
│         :8080            │
└──────────────────────────┘
```

## Subir o ambiente

```bash
docker compose up --build
```

## Serviços

| Serviço | Porta | Função |
|---|---:|---|
| API | 8000 | Inferência |
| Prometheus | 9090 | Coleta de métricas |
| Grafana | 3000 | Visualização |
| Airflow | 8080 | Treinamento/re-treinamento |

---

# 16. Otimização com ONNX

Como requisito de otimização de performance, foi utilizada a conversão do modelo para **ONNX** e execução através do **ONNX Runtime**.

Fluxo:

```text
Scikit-Learn
     │
     ▼
  skl2onnx
     │
     ▼
 model.onnx
     │
     ▼
ONNX Runtime
```

Conversão:

```bash
python training/convert_to_onnx.py
```

O script também valida o modelo convertido utilizando:

```python
onnx.checker.check_model(...)
```

---

# 17. Validação do modelo ONNX

A equivalência das previsões foi validada através de:

```bash
python training/test_onnx.py
```

O teste compara:

```text
Scikit-Learn
     ×
ONNX Runtime
```

Resultado validado:

```text
Scikit-Learn prediction: 4
ONNX Runtime prediction: 4

As previsões são iguais!
```

Isso demonstra que a conversão preservou a classe prevista no caso testado.

---

# 18. Benchmark de Latência

O benchmark foi executado com:

```text
1000 iterações
50 warmup
```

Resultados:

| Modelo | Latência média |
|---|---:|
| Scikit-Learn | 0.5188 ms |
| ONNX Runtime | 0.1042 ms |

Redução observada:

```text
79.91%
```

O benchmark foi realizado diretamente sobre a inferência do modelo, sem incluir a latência de rede/HTTP da API.

Execução:

```bash
python training/benchmark_latency.py
```

---

# 19. Benchmark de Throughput

Também foi realizado benchmark de throughput.

Configuração:

```text
10.000 iterações
100 warmup
```

Resultados:

| Modelo | Throughput |
|---|---:|
| Scikit-Learn | 1.987,79 inferências/s |
| ONNX Runtime | 7.497,23 inferências/s |

Aumento observado:

```text
277,16%
```

Equivalente a aproximadamente:

```text
3,77x
```

Execução:

```bash
python training/benchmark_throughput.py
```

Assim como o benchmark de latência, essa medição representa inferência direta do modelo e não throughput HTTP da API.

---

# 20. Requisitos do Tech Challenge

| Requisito | Implementação |
|---|---|
| Repositório GitHub | Projeto versionado |
| API REST | FastAPI |
| Modelo ML | TF-IDF + Logistic Regression |
| Dockerfile | Implementado |
| Docker Compose | Implementado |
| CI/CD | GitHub Actions |
| Lint | Ruff |
| Testes | Pytest |
| Airflow | DAG de treinamento |
| Monitoramento | Prometheus |
| Dashboard | Grafana |
| Métricas | Requests, status e latência |
| Otimização | ONNX + ONNX Runtime |
| Benchmark | Latência e throughput |
| Documentação | README |

---

# 21. Requisitos Não Funcionais

## Performance

A solução busca baixa latência de inferência através de:

- modelo leve;
- limitação do vocabulário TF-IDF;
- ONNX Runtime;
- benchmark comparativo.

## Observabilidade

A API possui métricas expostas através de:

```text
/metrics
```

permitindo acompanhamento através do Prometheus e Grafana.

## Escalabilidade

A API é stateless do ponto de vista da requisição, permitindo que múltiplas instâncias sejam executadas atrás de um balanceador em uma futura implantação cloud.

## Reprodutibilidade

O projeto utiliza:

- `requirements.txt`;
- Docker;
- Docker Compose;
- GitHub Actions;
- Airflow;
- scripts de treinamento;
- scripts de benchmark.

## Qualidade

São utilizados:

- Ruff;
- Pytest;
- versionamento Git;
- separação entre aplicação, treinamento, testes e monitoramento.

---

# 22. Como executar localmente

## Pré-requisitos

Instalar:

- Python 3.11;
- Git;
- Docker Desktop.

## Criar ambiente virtual

Windows:

```powershell
python -m venv .venv
```

Ativar:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Instalar dependências

```bash
pip install -r requirements.txt
```

## Treinar o modelo

```bash
python training/train.py
```

## Converter para ONNX

```bash
python training/convert_to_onnx.py
```

## Validar ONNX

```bash
python training/test_onnx.py
```

## Executar benchmarks

```bash
python training/benchmark_latency.py
python training/benchmark_throughput.py
```

## Executar testes

```bash
pytest -v
```

## Executar lint

```bash
ruff check .
```

## Executar API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

# 23. Executar com Docker Compose

Subir todos os serviços:

```bash
docker compose up --build
```

Acessos:

```text
API
http://localhost:8000

Swagger
http://localhost:8000/docs

Prometheus
http://localhost:9090

Grafana
http://localhost:3000

Airflow
http://localhost:8080
```

---

# 24. Demonstração recomendada

Para apresentação do Tech Challenge, recomenda-se demonstrar o fluxo:

```text
1. Apresentação do problema
        ↓
2. Dataset
        ↓
3. Arquitetura
        ↓
4. Treinamento do modelo
        ↓
5. API FastAPI
        ↓
6. Docker
        ↓
7. GitHub Actions
        ↓
8. Airflow
        ↓
9. Prometheus
        ↓
10. Grafana
        ↓
11. ONNX
        ↓
12. Benchmark de latência
        ↓
13. Benchmark de throughput
```

Durante a demonstração da API:

```http
POST /predict
```

com um texto médico de exemplo.

Em seguida:

```text
/metrics
```

para demonstrar a instrumentação.

Depois:

```text
Prometheus → Grafana
```

para demonstrar a observabilidade.

Finalmente:

```text
Scikit-Learn × ONNX Runtime
```

para demonstrar a otimização.

---

# 25. Critérios de avaliação × evidências

| Critério | Evidência no projeto |
|---|---|
| Modelo / Otimização | TF-IDF + Logistic Regression + ONNX Runtime |
| CI/CD | `.github/workflows/ci.yml` |
| Airflow | `airflow/dags/medical_triage_training.py` |
| Monitoramento | `prometheus-client` + Prometheus + Grafana |
| README | Este documento |
| Vídeo STAR | Demonstração do fluxo técnico |

---

# 26. Status do projeto

## Implementado

- API REST com FastAPI;
- Endpoint `/health`;
- Endpoint `/predict`;
- Endpoint `/metrics`;
- Modelo TF-IDF + Logistic Regression;
- Dataset Medical Abstracts TC Corpus;
- Treinamento automatizado por script;
- Testes automatizados;
- Lint com Ruff;
- Dockerfile;
- Docker Compose;
- Airflow;
- Prometheus;
- Grafana;
- Conversão para ONNX;
- Validação do ONNX;
- Benchmark de latência;
- Benchmark de throughput;
- Pipeline GitHub Actions.

## Em evolução

- Integração definitiva do ONNX Runtime ao caminho principal de inferência da API;
- Exportação/versão do dashboard Grafana;
- Benchmark de latência HTTP end-to-end;
- Integração da conversão ONNX ao fluxo de treinamento do Airflow;
- Deploy em cloud;
- Utilização de dataset com rótulos clínicos de urgência para uma aplicação de triagem real.

---

# 27. Considerações sobre produção

A arquitetura foi construída para demonstrar o ciclo de vida de um modelo de Machine Learning.

Para uma implantação clínica real seriam necessários, além dos componentes apresentados:

- dataset representativo e validado clinicamente;
- definição formal dos níveis de urgência;
- validação clínica;
- controle de acesso;
- proteção de dados sensíveis;
- auditoria;
- versionamento de modelos;
- monitoramento de drift;
- estratégia de rollback;
- validação de segurança;
- governança de Machine Learning.

Portanto, o projeto possui finalidade **acadêmica e demonstrativa**, não sendo destinado à tomada de decisão clínica.

---

# 28. Licença

Projeto acadêmico desenvolvido para o:

**Tech Challenge – Deploy de Modelo em Produção com Pipeline CI/CD, Monitoramento e Otimização de Latência.**

Uso destinado à avaliação acadêmica e demonstração dos conceitos de:

- Machine Learning;
- MLOps;
- CI/CD;
- Docker;
- Airflow;
- Observabilidade;
- Otimização de inferência.
