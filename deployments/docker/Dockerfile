FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace/plasformers
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY plasformers ./plasformers
COPY plas ./plas
COPY configs ./configs
COPY scripts ./scripts

RUN pip install --no-cache-dir -e ".[serve,export,ui]"

EXPOSE 8000 7860
CMD ["plas", "serve", "--host", "0.0.0.0", "--port", "8000"]
