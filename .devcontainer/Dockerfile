# --- Stage 1: Builder ---
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV DCNN_MODEL_GDRIVE_ID=16LFMIEADO3XI8uNqiUoKKlrzWlc1_Q-p
ENV DCNN_MODEL_DIR=dcnn_trained
RUN gdown ${DCNN_MODEL_GDRIVE_ID} -O ${DCNN_MODEL_DIR}.zip && \
    unzip ${DCNN_MODEL_DIR}.zip -d ${DCNN_MODEL_DIR} && \
    rm ${DCNN_MODEL_DIR}.zip

COPY pyproject.toml .
COPY scilink/ ./scilink/

# Runtime image should contain a non-editable install with extras
RUN pip install --no-cache-dir ".[all]" && \
    pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git && \
    pip install SciFiReaders

# --- Stage 2: Shared base ---
FROM python:3.12-slim AS base

ARG USERNAME=scilinkuser
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    openssh-client \
    procps \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} \
    && mkdir -p /workspaces /home/${USERNAME}/.vscode-server \
    && chown -R ${USERNAME}:${USERNAME} /workspaces /home/${USERNAME} \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && chmod 1777 /tmp

ENV HOME=/home/${USERNAME}
WORKDIR /workspaces

COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/scilink /usr/local/bin/scilink
COPY --from=builder /usr/local/bin/scifireader* /usr/local/bin/
COPY --from=builder /app/dcnn_trained /opt/scilink/dcnn_trained

ENV DCNN_MODEL_DIR=/opt/scilink/dcnn_trained
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# --- Stage 3a: Dev image ---
FROM base AS dev

USER root

RUN rm -r /usr/local/lib/python3.12/site-packages/scilink*

# install Claude CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get update && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && rm -rf /var/lib/apt/lists/*

# install github CLI
RUN (type -p wget >/dev/null || (sudo apt update && sudo apt install wget -y)) \
	&& sudo mkdir -p -m 755 /etc/apt/keyrings \
	&& out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
	&& cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& sudo mkdir -p -m 755 /etc/apt/sources.list.d \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& sudo apt update \
	&& sudo apt install gh -y


ARG USERNAME=scilinkuser
USER ${USERNAME}
WORKDIR /workspaces
CMD ["sleep", "infinity"]

# --- Stage 3b: Runtime image ---
FROM base AS runtime

ARG USERNAME=scilinkuser
USER ${USERNAME}
WORKDIR /home/${USERNAME}
ENTRYPOINT ["scilink"]
CMD ["--help"]
