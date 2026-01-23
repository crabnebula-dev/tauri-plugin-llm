FROM docker.io/debian:latest

LABEL maintainer="CrabNebula"
LABEL description="E2E testing container for Tauri apps with tauri-driver"

ENV DEBIAN_FRONTEND="noninteractive"
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US  
ENV LC_ALL=en_US.UTF-8
ENV DISPLAY=:99
ENV CARGO_HOME="/usr/local/cargo"
ENV RUSTUP_HOME="/usr/local/rustup"
ENV PATH="/usr/local/cargo/bin:$PATH"

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y build-essential curl git pkg-config ca-certificates

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
    task

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
RUN npm install -g pnpm
RUN cargo install tauri-driver --locked

RUN apt-get install -y locales-all locales
RUN update-locale

RUN printf "crabnebula-testing" > /etc/hostname

WORKDIR /testing

# Default entrypoint: run tests with virtual display
# ENTRYPOINT ["xvfb-run", "--auto-servernum", "--server-args=-screen 0 1280x720x24"]
# CMD ["task", "test:e2e:run"]
CMD ["/bin/bash"]