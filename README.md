# Identificação de Cone de Tráfego 
Tarefa: Desenvolver um classificador de imagens, utilizando VGG-16 para classificar o objeto de interesse (a ser definido pela dupla), que possa ser justificado como algo que um robô precise classificar por visão computacional. 

## Autores

- [João Henrique](https://github.com/joao4xz)
- [Marcelle Andrade ](https://github.com/Marcelleap)

## Informações referênciais 
- Disciplina: Robôs, Sensores e Aplicações
- Curso: Ciência da Computação - Campus Poços de Caldas - PPC - Noite - 2024/1
- Instituição: Pontifícia Universidade Católica de Minas Gerais - PUC Minas
- Professor: Harison Herman Silva
- Código da Disciplina: SGA_53255_55_2024_1_379100
- Formato: Graduação Presencial Síncrona - 2024/1


## Introdução
 
Esse projeto visa a criação de aprendizado de máquina, ultilizando como referencial a arquitetura VGG-16, capaz de detectar cones de trânsito nas imagens. O projeto inclui scripts para treinamento do modelo e para fazer previsões em novas imagens.

## Justificativa 
### Motivação

Em ambientes urbanos, os veículos autônomos enfrentam uma variedade de obstáculos, e muitas vezes a sinalização é feita através de cones de trânsito. Estes cones são utilizados para sinalizar desvios temporários, áreas de construção e outras obstruções. Portanto, o reconhecimento e a classificação precisos dos cones de trânsito são essenciais para a operação segura de carros autônomos.

Este trabalho justifica a necessidade de desenvolver um sistema de classificação de cones de trânsito utilizando a arquitetura VGG-16. O modelo será treinado para reconhecer cones em diversas condições, permitindo que, posteriormente, com a integração de sensores adicionais, ele possa ser utilizado em veículos autônomos. Isso garantirá uma navegação segura e eficiente, bem como a tomada de decisões informadas em tempo real, contribuindo significativamente para a operação segura e confiável de veículos autônomos em ambientes urbanos.

### Aplicações 

- Veículos Autônomos: Navegação segura em áreas urbanas.
- Robôs de Entrega: Identificação de obstáculos temporários.
- Sistemas de Segurança de Trânsito: Monitoramento e gestão de 
- Áreas de construção e manutenção.

### Impacto
- Melhora a segurança de operação.
- Aumenta a eficiência da navegação.
- Reduz riscos de acidentes.

## Objetivos 
Este projeto visa a criação de um modelo de aprendizado de máquina, utilizando como referencial a arquitetura VGG-16, capaz de detectar cones de trânsito em imagens. O projeto inclui scripts para treinamento do modelo e para fazer previsões em novas imagens.

## Metodologia 
- Visa a criação de um modelo de aprendizado de máquina, utilizando como referencial a arquitetura VGG-16, capaz de detectar cones de trânsito em imagens. O projeto inclui scripts para treinamento do modelo e para fazer previsões em novas imagens.

## Etapas do Desenvolvimento 
- Coleta de dados;
- Pré-processamento;
- Construção do modelo; 
- Treinamento e valição. 


## Instalação

### Pré- Requisito 
Certifique-se de ter os seguintes itens instalados 
- Python 3.x
- TensorFlow
- Keras
- Matplotlib
- NumPy

Para instalar os pacotes necessários pode-se usar o comando pip:
```bash
pip install tensorflow keras matplotlib numpy

```
### Treinando o modelo 

1. Clone o repositório: 
```bash
git clone https://github.com/joao4xz/traffic_cone_detection.git
cd traffic-cone-detection

```

2. Execute o script:
```bash
python train_model.py
```

3. O modelo treinado será salvo como 'traffic_cone_detector.h5'.

### Fazendo classificações:

1. Use o script predict_image.py para fazer previsões em novas imagens:
```bash
# predict_image.py

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

def load_and_preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

def predict_image(img_path, model):
    img = load_and_preprocess_image(img_path)
    prediction = model.predict(img)
    if prediction[0] > 0.5:
        print("Traffic Cone")
    else:
        print("Not a Traffic Cone")

# Load the model
model = load_model('traffic_cone_detector.h5')

# Path to the image you want to test
test_image_path = './test/ok10.png'  # Update with your test image path

# Predict the image
predict_image(test_image_path, model)
```

## Demonstração

### Video VGG-16 (Treinamento) 
### Video VGG-16 (Código)
### Imagens de Reprodução 
<img src="Fotos/WhatsApp%20Image%202024-06-08%20at%2022.36.31.jpeg" alt="Treinamento em andamento">
<img src="Fotos/WhatsApp Image 2024-06-09 at 06.45.13 (1).jpeg" alt="Treinamento e validação da acuracia">
<img src="Fotos/WhatsApp Image 2024-06-09 at 06.45.13.jpeg" alt="Treinamento e validação de perda">




## Referência

 - [TraCon: A novel dataset for real-time traffic cones detection using deep learning](https://github.com/ikatsamenis/Cone-Detection)
 - [Image Classification With VGG16](https://www.youtube.com/watch?si=mI8CS-0aA7wptte5&v=kJreyh5Gh8c&feature=youtu.be)
-  [ikatsamenis](https://github.com/ikatsamenis). You can find the original dataset in their repository. 
 
## Apêndice

[To view in English, click here.](English_REDME.md)