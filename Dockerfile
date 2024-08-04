FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /stage

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git
COPY requirements.txt /stage
RUN pip install -r requirements.txt
RUN pip install transformers==4.43.0

# Copy all files
COPY . /stage
