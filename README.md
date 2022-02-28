# Product Attribute Extraction

In the e-commerce world, extracting product attributes is important. The extraction of attribute labels and values from free-text product descriptions can be useful for many tasks, such as product matching, product categorization, faceted product search, and product recommendation. 

I use the BERT for token classification model to extract multi-attributes from the product offers in Indonesian e-commerce platform. The dataset is obtained from [this previous work](http://www.scielo.org.mx/scielo.php?script=sci_arttext&pid=S1405-55462018000401367#fn3). There are 16 kinds of attributes in their annotation scheme. Please check the paper directly to get more information about the dataset. 

## Prepare a Docker environment for the experiment
```
docker run --rm -it --name=attribute-extraction --gpus '"device=0"' --shm-size 32G -it --mount type=bind,src=<absolute path to product-attribute-extraction folder>,dst=/workspace/   pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

pip install -r requirements.txt
```
Alternatively, you can also use virtual environment.




