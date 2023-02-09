FROM continuumio/miniconda3
RUN conda install -y jax
RUN pip install vbjax
CMD python3 -m vbjax.tests