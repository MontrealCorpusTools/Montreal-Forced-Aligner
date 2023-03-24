FROM condaforge/mambaforge:22.11.1-4 as build

COPY ci/docker_environment.yaml .
RUN mkdir -p /mfa
RUN mamba env create -p /env -f docker_environment.yaml && conda clean -afy

COPY . /pkg
RUN conda run -p /env python -m pip install --no-deps /pkg

RUN useradd -ms /bin/bash mfauser
RUN chown -R mfauser /mfa
RUN chown -R mfauser /env
USER mfauser
ENV MFA_ROOT_ENVIRONMENT_VARIABLE=/mfa
RUN conda run -p /env mfa server init

RUN echo "source activate /env && mfa server start" > ~/.bashrc
ENV PATH /env/bin:$PATH
