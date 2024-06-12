FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

WORKDIR /code
COPY . .

RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install nano -y

RUN pip install --upgrade pip
RUN pip install addict
RUN pip install wandb
RUN pip install scikit-learn
RUN pip install pandas