FROM debian:bullseye

RUN apt-get update && \ 
apt-get install -y g++ wget bzip2 libarchive-tools && \
apt-get clean && \ 
rm -rf /var/lib/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda/ && \
    rm miniconda.sh

WORKDIR /home

ENV PATH=/opt/conda/bin:$PATH

COPY comp_env.yml .
RUN conda install -y -c conda-forge mamba && \
    mamba env create -f comp_env.yml && \ 
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc && \
    echo "conda activate comp" >> /etc/bash.bashrc

CMD ["/bin/bash","-l"]
