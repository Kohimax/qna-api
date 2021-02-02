# qna-api
Its complete rest API for Question and Answering with any unstructure document. The API is developed using haystack framework. The following features are supported:
1) Upload document to Elastic Document storage
2) Pretrain and Trained QNA

# Installation
`docker build -t qna:v1 .`

`docker run — name qna_app -d -p 8777:8777 xxxxxxxxx`

#### Note : xxxxxxxxx is the image id

# Training Example
```python
from haystack.reader.farm import FARMReader

#input directory of the labels answers.json file
train_data = "/usr/src/app/data/squad20"
# output directory of the model
train_model = "/usr/src/app/data/train_model"

reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=False)

reader.train(
    data_dir=train_data,
    train_filename="answers.json",
    n_epochs=10,
    dev_split = 0,
    save_dir=train_model)

print('Training successfully completed')
```
# Example
```python
document_store = ElasticsearchDocumentStore(host=app.config["host"],
                                                port=app.config["port"],
                                                username=app.config["username"],
                                                password=app.config["password"],
                                                index='electrical')
retriever = DensePassageRetriever(document_store=document_store, embedding_model="dpr-bert-base-nq",do_lower_case=True, use_gpu=False)
reader = FARMReader(model_name_or_path=app.config["train_model"] ,use_gpu=False)
finder = Finder(reader, retriever)
n = 1
question="asked your query here?"
prediction = finder.get_answers(question=question, top_k_retriever=10, top_k_reader=n)
```
# Code Explaination
The complete step by step explaination are provided [here](https://medium.com/analytics-vidhya/how-to-create-your-question-and-answering-flask-api-using-haystack-e97205a240d1)

