FROM docker.io/debian:latest

LABEL maintainer="CrabNebula"
LABEL description="E2E testing container for Tauri apps with tauri-driver"

ENV DEBIAN_FRONTEND="noninteractive"
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US  
ENV LC_ALL=en_US.UTF-8
ENV DISPLAY=0
ENV CARGO_HOME="/usr/local/cargo"
ENV RUSTUP_HOME="/usr/local/rustup"
ENV PATH="/usr/local/cargo/bin:/opt/go/latest/:/root/.local/bin:$PATH"

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y build-essential curl git pkg-config ca-certificates openssl libssl-dev
RUN curl -fLs https://deb.nodesource.com/setup_20.x | bash -
RUN curl -fLs 'https://dl.cloudsmith.io/public/task/task/setup.deb.sh' | bash
RUN apt-get update

RUN apt-get install -y --no-install-recommends \
    libwebkit2gtk-4.1-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev \
    webkit2gtk-driver \
    xvfb \
    xauth \
    nodejs \
    task \
    clang

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN cargo install tauri-driver --locked
RUN npm install -g pnpm

RUN apt-get install -y locales-all locales
RUN update-locale

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install python
RUN uv python install

# Install Huggingface CLI
RUN uv venv && . .venv/bin/activate && curl -LsSf https://hf.co/cli/install.sh | bash

RUN printf "crabnebula-testing" > /etc/hostname

WORKDIR /testing

ENTRYPOINT ["xvfb-run","--auto-servernum","--server-args='-screen 0 1280x720x24'"]