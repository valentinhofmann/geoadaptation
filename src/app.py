from flask import Flask, jsonify, request
from model_geoadaptation import GeoBertForMaskedLM, GeoElectraForMaskedLM
import os
import torch
import traceback
import sys
from helpers_zeroshot_geolocation import CollatorForZeroShot, DatasetForZeroShot
from helpers_geoadaptation import CollatorForMaskedLM, DatasetForMaskedLM
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

cities = { "bcms" : ["Bar", "Beograd", "Bor", "Dubrovnik", "Kragujevac", "Niš", "Podgorica", "Pula", "Rijeka", "Sarajevo", "Split", "Tuzla", "Zagreb", "Zenica"], 
           "scandi" : [], 
           "german" : [] }

scalers = { "bcms" : None} 

modelpaths = { "bcms" : "models/bcms_uncertainty_masked_25.torch", "scandi" : None, "german" : None}
pretrained_paths = { "bcms" : "classla/bcms-bertic", "scandi" : None, "german" : None}

models = { "bcms" : None, "scandi" : None, "german" : None}
tokenizers = { "bcms" : None, "scandi" : None, "german" : None}
models_geoad = { "bcms" : None, "scandi" : None, "german" : None}

device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

@app.route('/')
def index():
    return "Hello Geoworld"

@app.route('/load', methods=['POST'])
def load_model():
    reqjson = request.get_json()
    if not models[reqjson["region"]]:
        try:
            if not scalers[reqjson["region"]]:
                scalers[reqjson["region"]] = StandardScaler()
                if reqjson["region"] == "bcms": 
                    # computed the scaler from coordinates of four capitals (Zagreb, Beograd, Sarajevo, Podgorica) for now; but we should get the actual scaler from the training data
                    scalers[reqjson["region"]].fit(list(zip([15.96, 19.26, 18.38, 20.45], [45.84, 42.44, 43.85, 44.81])))

            if not tokenizers[reqjson["region"]]:
                tokenizers[reqjson["region"]] = AutoTokenizer.from_pretrained(pretrained_paths[reqjson["region"]], model_max_length=512)

            if reqjson["region"] == "bcms":
                models[reqjson["region"]] = GeoElectraForMaskedLM.from_pretrained(pretrained_paths[reqjson["region"]], "uncertainty", "masked", 2)
            else:
                models[reqjson["region"]] = GeoBertForMaskedLM.from_pretrained(pretrained_paths[reqjson["region"]], "uncertainty", "masked", 2)

            models[reqjson["region"]].load_state_dict(torch.load(os.path.join(app.root_path, modelpaths[reqjson["region"]]), map_location=device))
            models[reqjson["region"]] = models[reqjson["region"]].to(device)

            resp = jsonify(status='model {} successfully loaded'.format(reqjson["region"]))
            resp.status_code = 200
            return resp

        except Exception as ex:
            ex_type, ex_value, ex_traceback = sys.exc_info() 
            resp = jsonify(status='Exception occured: ' + str(ex_type.__name__) + ": " + str(ex_value))
            resp.status_code = 500
            return resp

@app.route('/predict', methods=['POST'])
def predict():
    reqjson = request.get_json()
    text = reqjson["text"]
    region = reqjson["region"]
    prompt = reqjson["prompt"] if "prompt" in reqjson else None

    sendback = {"region" : region, "text" : text }
    if prompt:
        sendback["prompt"] = prompt

    if not prompt:
        prompt = ""
    
    # zero-shot city prediction
    class_probs = get_zero_shot_city_probs(text, region, prompt)
    sendback["cities_pred"] = class_probs
    
    # longitude latitude prediction
    coords = get_coordinates(text, region)
    sendback["geoloc_pred"] = { "longitude" : coords[0][0], "latitude" : coords[0][1]}
    
    resp = jsonify(sendback)
    resp.status_code = 200
    return resp

def get_coordinates(text, region):
    data = type('',(object,),{"text": [text], "longitude" : [10.0], "latitude" : [10.0]})()
    dataset = DatasetForMaskedLM(data)
    collator = CollatorForMaskedLM(tokenizers[region], None)
    data_loader = DataLoader(dataset, batch_size=1, collate_fn=collator)

    longit = 0 
    latit = 0
    with torch.no_grad(): 
        for i, (batch_tensors, mlm_labels, points) in enumerate(data_loader):
            if i > 0:
                raise ValueError("There should be only one batch with only one instance")
            
            input_ids = batch_tensors['input_ids'].to(device)
            attention_mask = batch_tensors['attention_mask'].to(device)
            token_type_ids = batch_tensors['token_type_ids'].to(device)
            mlm_labels = mlm_labels.to(device)
            points = points.to(device)

            was_masked = False
            if models[region].head == "masked":
                was_masked = True
                models[region].head = None

            _, _, preds = models[region](
                input_ids,
                attention_mask,
                token_type_ids,
                mlm_labels,
                points,
                True
            )

            if was_masked:
                was_masked = False
                models[region].head = "masked"

            geopred = np.mean(preds, axis = 1)
            longit = geopred[0][0]
            latit = geopred[0][1]

            geodim = scalers[region].inverse_transform([[longit, latit]])
        
        return geodim

def get_zero_shot_city_probs(text, region, prompt):
    data = type('',(object,),{"text": [text], "location" : [cities[region][0]]})()
    dataset = DatasetForZeroShot(data)
    collator = CollatorForZeroShot(tokenizers[region], data, cities[region], prompt)
    data_loader = DataLoader(dataset, batch_size=1, collate_fn=collator)

    with torch.no_grad():
        class_scores = {}
        for i, (batch_tensors, masked, classes) in enumerate(data_loader):
            if i > 0:
                raise ValueError("There should be only one batch with only one instance")
            

            input_ids = batch_tensors['input_ids'].to(device)
            attention_mask = batch_tensors['attention_mask'].to(device)
            token_type_ids = batch_tensors['token_type_ids'].to(device)
            masked = masked.to(device)
            
            pred = models[region].get_preds(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    masked
            )
            
            if len(pred) != 1:
                raise ValueError("There should be only one batch with only one instance")
                
            class_scores = {tokenizers[region].convert_ids_to_tokens(c) : np.exp(pred[0][c].item()) for c in collator.classes}
            sum_all = np.sum(list(class_scores.values()))
            for c in class_scores:
                class_scores[c] = class_scores[c]/sum_all

        return class_scores


if __name__ == "__main__":
    app.run(debug=True)