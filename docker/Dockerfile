# Use Ubuntu 24.04 as the base image
FROM ubuntu:24.04

# Define build-time arguments for user, git username, git email, sudo password, user ID, and group ID
ARG USER_NAME
ARG GIT_USER_NAME
ARG GIT_USER_EMAIL
ARG SUDO_PASSWORD
ARG UID
ARG GID

# Set environment variables to ensure non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive

# Install sudo and other necessary utilities
RUN apt update && \
    apt upgrade -y && \
    apt install sudo -y

# Check if a user with the specified UID already exists; if so, delete the user
RUN if id -u $UID &>/dev/null; then \
        USER_TO_REMOVE=$(getent passwd $UID | cut -d: -f1) && \
        userdel -r $USER_TO_REMOVE; \
    fi

# Create the group with the specified GID
RUN groupadd -g $GID $USER_NAME

# Create the user with the specified UID, GID, and set up sudo access
RUN useradd -r -u $UID -g $GID -m $USER_NAME && \
    usermod -aG sudo $USER_NAME && \
    echo "$USER_NAME:$SUDO_PASSWORD" | chpasswd && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USER_NAME

# Set the user to the one specified by the build argument
USER $USER_NAME

# Update and upgrade apt and install different utilities
RUN sudo apt update && \
    sudo apt upgrade -y && \
    sudo apt install wget -y && \
    sudo apt install git-all -y

# Install Python dependencies
RUN sudo apt-get install software-properties-common -y && \
    sudo apt update && \
    sudo apt install build-essential -y && \
    sudo apt install libssl-dev libffi-dev python3-dev -y && \
    sudo apt install zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev -y && \
    sudo add-apt-repository ppa:deadsnakes/ppa && \
    sudo apt update && \
    sudo apt install python3.10 python3.10-distutils -y && \
    sudo wget https://bootstrap.pypa.io/get-pip.py && \
    sudo python3.10 get-pip.py

# Install Poetry
RUN sudo pip install poetry cffi
RUN poetry config virtualenvs.in-project true

# Set the working directory inside the container as a directory where the poetry.lock and the virtual environment will be created
WORKDIR /venv

# Copy the pyproject.toml and poetry.lock files into the container
COPY pyproject.toml poetry.lock* /venv

# Install dependencies from pyproject.toml using Poetry
RUN poetry install --no-root

# Initialize git user name and git user email
RUN git config --global user.email $GIT_USER_EMAIL
RUN git config --global user.name $GIT_USER_NAME

# Add the /app directory as a safe git directory
RUN git config --global --add safe.directory /app

# Set the working directory inside the container as the new user's home
WORKDIR /home/$USER_NAME

# Set the entrypoint
ENTRYPOINT ["/bin/bash"]
