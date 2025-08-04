FROM condaforge/mambaforge:22.11.1-4 AS build

COPY ci/docker_environment.yaml .
RUN mkdir -p /mfa
RUN useradd -ms /bin/bash mfauser
RUN chown -R mfauser /mfa
COPY . /pkg
RUN mamba env create -p /env -f docker_environment.yaml && conda clean -afy && \
 chown -R mfauser /env
RUN conda run -p /env python -m pip install speechbrain && \
 conda run -p /env python -m pip install --no-deps /pkg
USER mfauser
ENV MFA_ROOT_DIR=/mfa
RUN conda run -p /env mfa server init

RUN echo "source activate /env && mfa server start" > ~/.bashrc
ENV PATH=/env/bin:$PATH
