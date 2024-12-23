FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_OFFLINE=1

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 6000

CMD ["python", "llm_server.py"]
