FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MINIGPT_CHECKPOINT=/app/checkpoints/final.pt
WORKDIR /app
RUN python -m pip install --no-cache-dir "torch>=2.4" --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml LICENSE README.md ./
COPY minigpt ./minigpt
RUN python -m pip install --no-cache-dir . && \
    useradd --create-home --uid 10001 minigpt && mkdir -p checkpoints data
USER minigpt
VOLUME ["/app/checkpoints", "/app/data"]
EXPOSE 8000
CMD ["python", "-m", "minigpt.serve", "--host", "0.0.0.0", "--port", "8000"]
