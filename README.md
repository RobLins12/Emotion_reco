# Projeto de Reconhecimento de Expressão Facial (FER)

Este projeto utiliza inteligência artificial, especificamente redes neurais convolucionais, para o reconhecimento de expressões faciais. A implementação foi feita em Python, utilizando as bibliotecas OpenCV e Keras.

## Visão Geral

O objetivo deste projeto é criar um modelo capaz de identificar diferentes expressões faciais a partir de imagens. As expressões faciais que o modelo pode reconhecer incluem:

- Felicidade
- Tristeza
- Raiva
- Surpresa
- Medo
- Nojo
- Neutro

## Estrutura do Projeto

- `models/`: Diretório para salvar os modelos treinados.
- `Imagens_para_teste/`: Imagens para testar o model.
- `model.ipynb`: Jupyter notebook para exploração e teste do modelo.
- `face_detect.py`: Codigo base para detectar faces.
- `emotion_reco.py`: Arquivo do programa em si.
- `requirements.txt`: Arquivo contendo as dependências do projeto.
- `DOCS.md` : Arquivo de documentação de código.
- `README.md`: Este arquivo.

## Requisitos

- Python 3.6+
- OpenCV
- Keras
- TensorFlow (backend para Keras)

## Instalando os requisitos

- Site para download da opencv: https://opencv.org/releases/
- Site para download do python: https://www.python.org/downloads/

## Comandos

Primeiro clone o projeto. Dentro do projeto faça os passos a seguir.

1 - Instale as dependências, para isso execute:

```bash
pip install -r requirements.txt
```

2 - No terminal rode o programa, para isso execute:

```bash
python emotion_reco.py
```
