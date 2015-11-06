FROM ubuntu:14.04
MAINTAINER Kevin James Matzen <kmatzen@cs.cornell.edu>

RUN apt-get update && \
    apt-get install libpython-dev python-numpy build-essential \
    libatlas-base-dev python-msgpack python-scipy git -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/tmp/* 

RUN git clone https://github.com/kmatzen/pydro.git

RUN cd pydro && \
    python setup.py build && \
    python setup.py install
