from transformers import BertForPreTraining

model = BertForPreTraining.from_pretrained('bert-base-uncased')

print(model)
