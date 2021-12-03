FROM continuumio/miniconda3
RUN conda create -n aligner -c conda-forge montreal-forced-aligner
