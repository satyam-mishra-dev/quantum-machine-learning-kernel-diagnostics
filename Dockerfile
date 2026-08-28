FROM python:3.12-slim

# HF Spaces runs as uid 1000 and expects the app on $PORT (7860 by default).
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH \
    HOME=/home/user \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /app
COPY --chown=user requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY --chown=user qnidaan/ qnidaan/
COPY --chown=user models/ models/
COPY --chown=user runs/ runs/

EXPOSE 7860
CMD ["uvicorn", "qnidaan.app:app", "--host", "0.0.0.0", "--port", "7860"]
