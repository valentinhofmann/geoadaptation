import argparse
import pickle
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from helpers_zeroshot_geolocation import *
from model_geoadaptation import *


def main():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument('--model', default=None, type=str, required=True, help='Name of model.')
    parser.add_argument('--data', default=None, type=str, required=True, help='Name of data.')
    parser.add_argument('--batch_size', default=None, type=int, required=True, help='Batch size.')
    parser.add_argument('--mtl', default=None, type=str, help='MTL method.')
    parser.add_argument('--head', default=None, type=str, help='Type of prediction head.')
    parser.add_argument('--pretrained', default=False, action='store_true', help='Use pretrained model.')
    parser.add_argument('--random', default=False, action='store_true', help='Use random prediction.')
    parser.add_argument('--location', default=None, type=str, help='Type of location.')
    parser.add_argument('--device', default=None, type=int, required=True, help='Selected CUDA device.')
    args = parser.parse_args()

    # Load data
    data = pd.read_json('../data/{}/zeroshot_{}.json'.format(args.data, args.location), lines=True)
    dataset = DatasetForZeroShot(data)

    # Print out training specifics and parameters
    print('Model: {}'.format(args.model))
    print('Data: {}'.format(args.data))
    print('Batch size: {:02d}'.format(args.batch_size))
    print('Location: {}'.format(args.location))
    if args.mtl:
        print('Using {} MTL...'.format(args.mtl))
    if args.head:
        print('Using {} prediction head...'.format(args.head))

    # Define filename
    filename = args.data
    if args.mtl:
        filename += '_{}'.format(args.mtl)
    if args.head:
        filename += '_{}'.format(args.head)
    if args.pretrained:
        filename += '_pretrained'
    if args.random:
        filename += '_random'

    if args.random:
        true = dataset.locations
        preds = random.sample(dataset.locations, len(dataset.locations))
        print(accuracy_score(true, preds))
        with open('preds_zeroshot/{}_{}.p'.format(filename, args.location), 'wb') as f:
            pickle.dump([true, preds], f)
        return 0

    best_acc = None

    for generation in range(1, 26):
        filename_in = filename + '_{:02d}'.format(generation)

        # Load tokenizer
        tok = AutoTokenizer.from_pretrained(args.model, model_max_length=512)

        # Load data loaders
        collator = CollatorForZeroShot(tok, args.data, dataset.locations)
        print(collator.classes)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)

        # Define dim of output layer
        if args.mtl:
            output_dim = 2
        else:
            output_dim = 0

        # Define device
        device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

        # Initialize model
        if args.model == 'classla/bcms-bertic':
            model = GeoElectraForMaskedLM.from_pretrained(args.model, args.mtl, args.head, output_dim)
        else:
            model = GeoBertForMaskedLM.from_pretrained(args.model, args.mtl, args.head, output_dim)
        if not args.pretrained:
            print('Loading model weights from {}...'.format(filename_in))
            model.load_state_dict(torch.load('trained/{}.torch'.format(filename_in), map_location=device))
        model = model.to(device)

        print('Evaluate model...')
        model.eval()
        true, preds = list(), list()
        with torch.no_grad():
            for i, (batch_tensors, masked, classes) in enumerate(data_loader):
                input_ids = batch_tensors['input_ids'].to(device)
                attention_mask = batch_tensors['attention_mask'].to(device)
                token_type_ids = batch_tensors['token_type_ids'].to(device)
                masked = masked.to(device)
                if i == 0:
                    print(input_ids[0, :])
                pred = model.get_preds(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    masked
                )
                true.extend(classes)
                for p in pred:
                    class_scores = [p[c].item() for c in collator.classes]
                    preds.append(collator.classes[np.argmax(class_scores)])

        if args.pretrained:
            print(accuracy_score(true, preds))
            with open('preds_zeroshot/{}_{}.p'.format(filename, args.location), 'wb') as f:
                pickle.dump([true, preds], f)
            return 0

        with open('results_zeroshot/{}_{}.txt'.format(filename, args.location), 'a+') as f:
            f.write('{:.3f}\n'.format(accuracy_score(true, preds)))

        if best_acc is None or accuracy_score(true, preds) > best_acc:
            best_acc = accuracy_score(true, preds)
            with open('preds_zeroshot/{}_{}.p'.format(filename, args.location), 'wb') as f:
                pickle.dump([true, preds], f)


if __name__ == '__main__':
    main()
