FROM ubuntu:22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    tshark \
    wireshark-common \
    make \
    binutils \
    xxd \
    && rm -rf /var/lib/apt/lists/*

# Create working directory with proper permissions
RUN mkdir -p /app/answers /app/exported_files && \
    chmod -R 777 /app

WORKDIR /app
USER nobody

COPY Makefile .

ENTRYPOINT ["make"]