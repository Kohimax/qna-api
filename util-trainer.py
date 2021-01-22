from haystack.reader.farm import FARMReader

#input directory of the labels answers.json file
train_data = "/usr/src/app/data/squad20"
# output directory of the model
train_model = "/usr/src/app/data/train_model"

reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=False)

reader.train(
    data_dir=train_data,
    train_filename="answers.json",
    n_epochs=20,
    dev_split = 0,
    save_dir=train_model)

print('Training successfully completed')