# Using official ubuntu image as a parent image
FROM ubuntu:latest

# Setting the working directory to /app
WORKDIR /Smore

# compile python from source - avoid unsupported library problems
RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

RUN apt-get install -y python3.10

RUN apt-get install --assume-yes --no-install-recommends --quiet \
    curl \
    gcc \
    build-essential \
    python3-dev\
    python3-pip \
    wget \
    unzip \
    && apt-get clean all

#RUN wget https://dotnetcli.azureedge.net/dotnet/Runtime/5.0.17/dotnet-runtime-5.0.17-linux-arm64.tar.gz -O dotnet.tar.gz \
#    && tar -xf dotnet.tar.gz \
#    && cp dotnet /usr/bin
RUN wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh \
    && chmod +x dotnet-install.sh \
    && ./dotnet-install.sh -i /usr/bin --channel 5.0

RUN wget https://www.openssl.org/source/openssl-1.1.1c.tar.gz \
    && tar -xzvf openssl-1.1.1c.tar.gz \
    && cd openssl-1.1.1c \
    && ./config \
    && make \
    && make install \
    && export export LD_LIBRARY_PATH="/usr/local/lib"

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

COPY requirements.txt /Smore

RUN pip install -r requirements.txt

RUN python3 -m spacy download en_core_web_md

# Copy the current directory contents into the container at /app
COPY . /Smore
RUN unzip cache_ae.zip

WORKDIR /Smore/eval_res
RUN unzip flashgpt_results.zip

WORKDIR /Smore
RUN cp "benchmarks/benchmarks.csv" "eval_res/ae_results.csv"

CMD ["/bin/bash"]