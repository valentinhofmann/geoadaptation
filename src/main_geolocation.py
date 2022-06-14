import argparse
import pickle
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from helpers_evaluation import *
from helpers_geolocation import *
from model_geolocation import *


def main():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument('--model', default=None, type=str, required=True, help='Name of model.')
    parser.add_argument('--data', default=None, type=str, required=True, help='Name of data.')
    parser.add_argument('--batch_size', default=None, type=int, required=True, help='Batch size.')
    parser.add_argument('--n_epochs', default=None, type=int, required=True, help='Number of epochs.')
    parser.add_argument('--lr', default=None, type=float, required=True, help='Learning rate.')
    parser.add_argument('--generation', default=None, type=int, help='Model generation.')
    parser.add_argument('--mtl', default=None, type=str, help='MTL method.')
    parser.add_argument('--head', default=None, type=str, help='Type of prediction head.')
    parser.add_argument('--geo_type', default=None, type=str, help='Geo type for prediction.')
    parser.add_argument('--pretrained', default=False, action='store_true', help='Use pretrained model.')
    parser.add_argument('--device', default=None, type=int, required=True, help='Selected CUDA device.')
    args = parser.parse_args()

    # Load data
    train = pd.read_json('../data/{0}/train_gp.json'.format(args.data), lines=True)
    train_dataset = DatasetForGeoprediction(train)
    dev = pd.read_json('../data/{0}/dev_gp.json'.format(args.data), lines=True)
    dev_dataset = DatasetForGeoprediction(dev, train_dataset.kmeans, train_dataset.scaler)
    test = pd.read_json('../data/{0}/test_gp.json'.format(args.data), lines=True)
    test_dataset = DatasetForGeoprediction(test, train_dataset.kmeans, train_dataset.scaler)

    # Print out training specifics and parameters
    print('Model: {}'.format(args.model))
    print('Data: {}'.format(args.data))
    print('Batch size: {:02d}'.format(args.batch_size))
    print('Number of epochs: {}'.format(args.n_epochs))
    print('Learning rate: {:.0e}'.format(args.lr))
    print('Geo type: {}'.format(args.geo_type))
    if args.mtl:
        print('Using {} MTL...'.format(args.mtl))
    if args.head:
        print('Using {} prediction head...'.format(args.head))

    # Define filenames
    filename_in = args.data
    if args.mtl:
        filename_in += '_{}'.format(args.mtl)
    if args.head:
        filename_in += '_{}'.format(args.head)
    if args.pretrained:
        filename_in += '_pretrained'
    else:
        filename_in += '_{}'.format(args.generation)
    filename_out = filename_in + '_{}'.format(args.geo_type)

    best_dist, *_ = get_best('results_geoprediction/{}.txt'.format(filename_out))
    print('Best distance so far: {}'.format(best_dist))

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, model_max_length=128)

    # Load data loaders
    cltr = CollatorForGeoprediction(tok)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=cltr, shuffle=True, drop_last=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=cltr)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=cltr)

    # Define dim of output layer
    if args.geo_type == 'kmeans':
        output_dim = 75
    elif args.geo_type == 'points':
        output_dim = 2

    # Define device
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    # Initialize model
    if args.model == 'classla/bcms-bertic':
        model = ElectraForGeoprediction(args.model, args.geo_type, output_dim)
        if not args.pretrained:
            print('Loading model weights from {}...'.format(filename_in))
            pretrained_dict = torch.load('trained/{}.torch'.format(filename_in), map_location=device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'electra' in k}
            model_dict = model.state_dict()
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    else:
        model = BertForGeoprediction(args.model, args.geo_type, output_dim)
        if not args.pretrained:
            print('Loading model weights from {}...'.format(filename_in))
            pretrained_dict = torch.load('trained/{}.torch'.format(filename_in), map_location=device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'bert' in k}
            model_dict = model.state_dict()
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    model = model.to(device)

    # Initialize optimizer with scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_warmup_steps = 3 * len(train_loader)
    num_training_steps = args.n_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Training loop
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    for epoch in range(1, args.n_epochs + 1):
        print('Train model...')
        model.train()
        for i, (batch_tensors, points, cluster_labels) in enumerate(train_loader):
            input_ids = batch_tensors['input_ids'].to(device)
            attention_mask = batch_tensors['attention_mask'].to(device)
            token_type_ids = batch_tensors['token_type_ids'].to(device)
            if i == 0:
                print(input_ids[0, :])
            points = points.to(device)
            cluster_labels = cluster_labels.to(device)
            optimizer.zero_grad()
            loss, preds = model(
                input_ids,
                attention_mask,
                token_type_ids,
                points,
                cluster_labels
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Evaluation
        print('Evaluate model...')
        model.eval()
        with torch.no_grad():
            dev_true = list()
            dev_preds = list()
            for batch_tensors, points, cluster_labels in dev_loader:
                input_ids = batch_tensors['input_ids'].to(device)
                attention_mask = batch_tensors['attention_mask'].to(device)
                token_type_ids = batch_tensors['token_type_ids'].to(device)
                points = points.to(device)
                cluster_labels = cluster_labels.to(device)
                loss, preds = model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    points,
                    cluster_labels
                )
                dev_true.extend(points.detach().cpu().tolist())
                if args.geo_type == 'kmeans':
                    preds = train_dataset.cluster_labels2points(preds)
                dev_preds.extend(preds)

            test_true = list()
            test_preds = list()
            for batch_tensors, points, cluster_labels in test_loader:
                input_ids = batch_tensors['input_ids'].to(device)
                attention_mask = batch_tensors['attention_mask'].to(device)
                token_type_ids = batch_tensors['token_type_ids'].to(device)
                points = points.to(device)
                cluster_labels = cluster_labels.to(device)
                loss, preds = model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    points,
                    cluster_labels
                )
                test_true.extend(points.detach().cpu().tolist())
                if args.geo_type == 'kmeans':
                    preds = train_dataset.cluster_labels2points(preds)
                test_preds.extend(preds)

        # Print and store results
        median_dist_dev = median_dist(dev_preds, dev_true, train_dataset.scaler, args.geo_type)
        median_dist_test = median_dist(test_preds, test_true, train_dataset.scaler, args.geo_type)
        mean_dist_dev = mean_dist(dev_preds, dev_true, train_dataset.scaler, args.geo_type)
        mean_dist_test = mean_dist(test_preds, test_true, train_dataset.scaler, args.geo_type)
        print(median_dist_dev, median_dist_test, mean_dist_dev, mean_dist_test, args.lr, epoch)
        with open('results_geoprediction/{}.txt'.format(filename_out), 'a+') as f:
            f.write('{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.0e}\t{}\n'.format(
                median_dist_dev,
                median_dist_test,
                mean_dist_dev,
                mean_dist_test,
                args.lr,
                epoch
            ))

        # Store predictions
        if best_dist is None or median_dist_dev < best_dist:
            best_dist = median_dist_dev
            with open('preds_geoprediction/{}.p'.format(filename_out), 'wb') as f:
                pickle.dump([dev_preds, dev_true, test_preds, test_true], f)


if __name__ == '__main__':
    main()
