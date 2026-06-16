FROM python:3.11-slim

WORKDIR /workspace/solution

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1

COPY requirements.txt /workspace/solution/requirements.txt
RUN pip install --no-cache-dir -r /workspace/solution/requirements.txt

COPY . /workspace/solution