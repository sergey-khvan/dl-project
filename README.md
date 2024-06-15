# Deep Learning Project

## Topic: DNA Sequences for Identifying Viral Genomes in Human Samples

## How to use:

1. Clone the repository 
```
git clone https://github.com/sergey-khvan/dl-project
```
2. Create the docker image 
```
docker build -t image-name . 
```
3. Run the docker container using the docker image 
```
docker run -d	--gpus '"device=0"' --name "container-name" image-name tail -f /dev/null 
```

4. Interact with your container
```
docker exec -it container-name /bin/bash
```
5. Run the code from the train directory 
```
cd train
python3 train.py
```


**Link to the paper:** [paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0222271)

**Original implementation:** [ViraMiner github](https://github.com/NeuroCSUT/ViraMiner)

