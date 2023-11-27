FROM nvcr.io/nvidia/pytorch:23.10-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    POETRY_HOME="/opt/poetry"
ENV PATH="${POETRY_HOME}/bin:${PATH}"

# Install poetry
RUN python -c 'from urllib.request import urlopen; print(urlopen("https://install.python-poetry.org").read().decode())' | python -
RUN poetry config virtualenvs.create false

# Install dependencies
WORKDIR /app
COPY pyproject.toml /app/
RUN poetry install --no-interaction --no-ansi --no-root -vvv

# COPY . /app/
