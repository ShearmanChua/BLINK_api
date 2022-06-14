FROM python:3.8

RUN apt-get update
RUN apt-get -y install python3-pip vim git

RUN pip install -U pip
RUN pip install -e git+https://github.com/facebookresearch/BLINK.git#egg=BLINK
RUN pip install fastapi pandas requests torch && pip install "uvicorn[standard]"

RUN mkdir /blink && mkdir /blink/models
COPY entity_linking_container/src/* /blink/
WORKDIR /blink

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]

EXPOSE 5000
