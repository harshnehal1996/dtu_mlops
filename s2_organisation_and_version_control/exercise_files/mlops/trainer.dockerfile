# BASE IMAGE
FROM python:3.7-slim

# install python
RUN apt update &&\
    apt install --no-install-recommends -y build-essential gcc &&\
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_model.py", "--lr=0.06"]

