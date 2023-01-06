FROM ubuntu:jammy

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y python3-pip python3-virtualenv
RUN pip install pipenv

ADD ./ src
RUN cd src && pipenv install -d
