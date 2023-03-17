FROM condaforge/mambaforge:4.9.2-5 as conda

COPY docker_environment.yaml .
RUN mkdir -p /mfa
RUN mamba env create -p /env -f docker_environment.yaml && conda clean -afy

COPY . /pkg
RUN conda run -p /env python -m pip install --no-deps /pkg

RUN useradd -ms /bin/bash mfauser
RUN chmod -R 775 /opt/conda
RUN chown -R mfauser /env
RUN chown -R mfauser /mfa
USER mfauser

ENV MFA_ROOT_ENVIRONMENT_VARIABLE=/mfa
RUN conda run -p /env mfa server init

FROM gcr.io/distroless/base-debian10

COPY --from=conda /env /env
COPY --from=conda /mfa /mfa

ENV MFA_ROOT_ENVIRONMENT_VARIABLE=/mfa

ENTRYPOINT ["/env/bin/python", "-m", "montreal_forced_aligner"]
