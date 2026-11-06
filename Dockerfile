FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[api,deep]'
COPY artifacts/model.joblib /app/artifacts/model.joblib
EXPOSE 8000
ENV MODEL_PATH=/app/artifacts/model.joblib
CMD ["uvicorn", "genosplice.api:app", "--host", "0.0.0.0", "--port", "8000"]
