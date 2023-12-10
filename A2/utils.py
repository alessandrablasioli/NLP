# pytorch
import torch


def model_bert_cps_predict(model, dataloader):
    model.eval()  

    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            data = batch

            X_1, X_2, stance = data["Premise"], data["Conclusion"], data["Stance"]

            encoded_1 = model.tokenizer(X_1, padding=True, truncation=True, return_tensors="pt")
            encoded_2 = model.tokenizer(X_2, padding=True, truncation=True, return_tensors="pt")

            batch_predictions = model(encoded_1, encoded_2, stance)
            predictions.append(batch_predictions)

    all_predictions = torch.cat(predictions)

    return all_predictions



def model_bert_cp_predict(model, dataloader):
    model.eval()  

    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            data = batch

            X_1, X_2 = data["Premise"], data["Conclusion"]

            encoded_1 = model.tokenizer(X_1, padding=True, truncation=True, return_tensors="pt")
            encoded_2 = model.tokenizer(X_2, padding=True, truncation=True, return_tensors="pt")

            batch_predictions = model(encoded_1, encoded_2)
            predictions.append(batch_predictions)

    all_predictions = torch.cat(predictions)

    return all_predictions


def model_bert_c_predict(model, dataloader):
    model.eval()  

    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            data = batch

            X_1 = data["Premise"]

            encoded_1 = model.tokenizer(X_1, padding=True, truncation=True, return_tensors="pt")

            batch_predictions = model(encoded_1)
            predictions.append(batch_predictions)

    all_predictions = torch.cat(predictions)

    return all_predictions


