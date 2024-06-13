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
- `Dockerfile`: Ambiente em docker configurado.
- `requirements.txt`: Arquivo contendo as dependências do projeto.
- `README.md`: Este arquivo.

## Requisitos

- Python 3.6+
- OpenCV
- Keras
- TensorFlow (backend para Keras)

Para instalar as dependências, execute:

```bash
pip install -r requirements.txt
```

Para construir um container e imagem, execute:

```bash
docker build -t "nome_da_imagem" .
```

Para rodar, execute:

```bash
docker run "nome_da_imagem"
```
