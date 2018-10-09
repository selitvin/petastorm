# Ubuntu based developer image
FROM ubuntu:16.04

RUN apt-get update

RUN apt-get install -y software-properties-common python3-software-properties python-software-properties
RUN add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update
RUN apt-get install -y \
                    git \
                    python3.6 \
                    python3.6-dev \
                    python3.6-venv \
                    build-essential \
                    libpq-dev \
                    libssl-dev \
                    openssl \
                    libffi-dev \
                    zlib1g-dev \
                    build-essential \
                    libpq-dev \
                    libssl-dev \
                    openssl \
                    libffi-dev \
                    zlib1g-dev \
                    python3-venv

# OpenCV dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
                libglib2.0 \
                libsm6 \
                libfontconfig1 \
                libxrender1


# Installing Java 8
RUN DEBIAN_FRONTEND='noninteractive' apt-get install -y \
                software-properties-common \
                python3-software-properties

RUN add-apt-repository ppa:webupd8team/java
RUN apt-get install -y openjdk-8-jdk

ADD bootstrap-dev-image-p36.sh /root/

CMD cd /root && ./bootstrap-dev-image-p36.sh