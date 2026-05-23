FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-base.txt requirements-ml.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-base.txt \
    && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install transformers sentencepiece

COPY app ./app
COPY scripts ./scripts
COPY models ./models
COPY data ./data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
