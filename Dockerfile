FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
WORKDIR /app

ENV PATH="/home/user/.local/bin:$PATH"

COPY --chown=user ./requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
ENV PYTHONPATH=/app/src
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

