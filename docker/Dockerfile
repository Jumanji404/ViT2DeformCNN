# Use Ubuntu 24.04 as the base image
FROM ubuntu:24.04

# Set environment variables to ensure non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive

# Update and upgrade apt and install different utilitaries
RUN apt update && \
    apt upgrade -y && \
    apt install wget -y

# Update the package list and install python dependencies
RUN apt-get install software-properties-common -y && \
    apt update && \
    apt install build-essential -y && \
    apt install libssl-dev libffi-dev python3-dev -y && \
    apt install zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install python3.10 python3.10-distutils -y && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py

# Install Poetry
RUN pip install poetry cffi
RUN poetry config virtualenvs.in-project true

# Set the working directory inside the container
WORKDIR /

# Copy the pyproject.toml and poetry.lock files into the container
COPY pyproject.toml poetry.lock* /

# Install dependencies from pyproject.toml using Poetry
RUN poetry install --no-root

ENTRYPOINT ["/bin/bash"]