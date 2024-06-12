FROM python:latest

WORKDIR /reconhecimento-de-emocoes

RUN apt update  && apt install -y git vim

COPY requirements.txt ./

RUN python -m pip install --upgrade pip  && python -m pip install -r requirements.txt

CMD python emotion_reco.py