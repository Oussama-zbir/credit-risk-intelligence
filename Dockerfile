# syntax=docker/dockerfile:1

# The scoring service, without a model. The artifact is mounted at runtime:
#
#     docker run -p 8000:8000 -v "$PWD/artifacts/<model_id>:/artifact:ro" credit-risk-service
#
# so one image serves any artifact, and a new model is a new mount, not a new
# build. The artifact refuses to load under a scikit-learn other than the one it
# was saved with, so the image installs the versions pinned in constraints.txt;
# train with the same constraints to get an artifact this image will serve.
# Python 3.12 matches the interpreter the published artifacts record.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CREDIT_RISK_ARTIFACT_DIR=/artifact

WORKDIR /app

COPY pyproject.toml README.md LICENSE constraints.txt ./
COPY src ./src
RUN pip install ".[service]" -c constraints.txt

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# A failed artifact check raises in the lifespan; with lifespan "on", uvicorn
# exits non-zero instead of serving without a model.
CMD ["uvicorn", "--factory", "credit_risk.service:app_from_env", \
     "--host", "0.0.0.0", "--port", "8000", "--lifespan", "on"]
