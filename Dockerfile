FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-core.txt /app/
RUN pip install --no-cache-dir -r requirements-core.txt pytest==8.3.5 'autogluon.tabular[lightgbm,catboost,xgboost]==1.3.1' matplotlib==3.9.4 seaborn==0.13.2
COPY automl_agent /app/automl_agent
COPY context_compressor /app/context_compressor
COPY ml_knowledge /app/ml_knowledge
COPY solution.py /app/solution.py
COPY data_for_participants/requirements.txt /app/requirements.txt
COPY tests /app/tests
COPY data_for_participants/requirements.txt /app/data_for_participants/requirements.txt
COPY knowledge_runtime /app/knowledge_runtime
COPY tools /app/tools
CMD ["python", "-m", "pytest", "tests", "-q", "--disable-warnings", "--tb=short"]
