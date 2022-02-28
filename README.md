# Product Attribute Extraction

In the e-commerce world, extracting product attributes is important. The extraction of attribute labels and values from free-text product descriptions can be useful for many tasks, such as product matching, product categorization, faceted product search, and product recommendation. 

The image below shows the extracted attribute of a product titled 'Spigen Samsung S9 Case Ultra Copper Gold'.

![Architecture](img/ecommerce-product-title-attribute.jpg)

I examine this problem as a sequence labelling task and utilize BERT for Token Classification model to extract multi-attributes from the product offers in Indonesian e-commerce platform. The dataset is obtained from [this previous work](http://www.scielo.org.mx/scielo.php?script=sci_arttext&pid=S1405-55462018000401367#fn3). There are 16 kinds of attributes in their annotation scheme. Please check the paper directly to get more information about the dataset. 

## Prepare a Docker environment for the experiment
```
docker run --rm -it --name=attribute-extraction --gpus '"device=0"' --shm-size 32G -it --mount type=bind,src=<absolute path to product-attribute-extraction folder>,dst=/workspace/   pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

pip install -r requirements.txt
```
Alternatively, you can also use virtual environment.

## Pipeline

![Architecture](img/ecommerce-product-attribute-extraction-architecture.jpg)

### Data Preparation

The dataset is annotated using the Enamex format. The preparation contains several steps, i.e., convert Enamex to Stanford and convert Stanford to BIO format. After performing a manual inspection on the dataset, I found that some incorrect labellings from the raw data cause a failure when converting the Enamex into Stanford format. To handle this, I manually fix the wrong Enamex format from the original file.

```
python preprocess.py
```

### Sequence Labelling

Sequence labeling is a typical NLP task that assigns a specific label or class to each token within a sequence. In this context, a single word is a 'token'. These tags can be used in further downstream models as features of the token, or to enhance the model. Fine-tuning BERT for text tagging applications is illustrated in the figure below.

![BERT For Token Classification Architecture](img/bert-for-token-classification.svg)

To fine tune the model, please run this script
```
python fine_tune.py
```




