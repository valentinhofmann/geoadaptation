import argparse
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.optimization import AdamW, get_constant_schedule_with_warmup

from helpers_evaluation import *
from helpers_geoadaptation import *
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
    parser.add_argument('--n_epochs', default=None, type=int, required=True, help='Number of epochs.')
    parser.add_argument('--lr', default=None, type=float, required=True, help='Learning rate.')
    parser.add_argument('--mtl', default=None, type=str, help='MTL method.')
    parser.add_argument('--head', default=None, type=str, help='Type of prediction head.')
    parser.add_argument('--device', default=None, type=int, required=True, help='Selected CUDA device.')
    args = parser.parse_args()

    # Load data
    train = pd.read_json('../data/{}/train_ga.json'.format(args.data), lines=True)
    train_dataset = DatasetForMaskedLM(train)
    dev = pd.read_json('../data/{}/dev_gp.json'.format(args.data), lines=True)
    dev_dataset = DatasetForMaskedLM(dev, train_dataset.scaler)
    test = pd.read_json('../data/{}/test_gp.json'.format(args.data), lines=True)
    test_dataset = DatasetForMaskedLM(test, train_dataset.scaler)

    # Print out training specifics and parameters
    print('Model: {}'.format(args.model))
    print('Data: {}'.format(args.data))
    print('Batch size: {:02d}'.format(args.batch_size))
    print('Number of epochs: {}'.format(args.n_epochs))
    print('Learning rate: {:.0e}'.format(args.lr))
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

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, model_max_length=128)

    # Load data loaders
    cltr = CollatorForMaskedLM(tok, args.head)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=cltr, shuffle=True, drop_last=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=cltr)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=cltr)

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
    model = model.to(device)

    # Initialize optimizer with scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_warmup_steps = 3 * len(train_loader)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)

    # Training loop
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    for epoch in range(1, args.n_epochs + 1):
        print('Train model...')
        model.train()
        for i, (batch_tensors, mlm_labels, points) in enumerate(train_loader):
            input_ids = batch_tensors['input_ids'].to(device)
            attention_mask = batch_tensors['attention_mask'].to(device)
            token_type_ids = batch_tensors['token_type_ids'].to(device)
            mlm_labels = mlm_labels.to(device)
            if i == 0:
                print(input_ids[0, :])
            points = points.to(device)
            optimizer.zero_grad()
            mlm_loss, geo_loss, preds = model(
                input_ids,
                attention_mask,
                token_type_ids,
                mlm_labels,
                points,
                False
            )
            if args.mtl:
                loss = mlm_loss + geo_loss
                loss.backward()
            else:
                mlm_loss.backward()
            optimizer.step()
            scheduler.step()

        # Evaluation
        print('Evaluate model...')
        model.eval()
        with torch.no_grad():
            if args.mtl == 'uncertainty':
                print(torch.exp(-model.etas[0]), torch.exp(-model.etas[1]))
            dev_mlm_losses = list()
            if args.mtl:
                dev_true = list()
                dev_preds = list()
            for batch_tensors, mlm_labels, points in dev_loader:
                input_ids = batch_tensors['input_ids'].to(device)
                attention_mask = batch_tensors['attention_mask'].to(device)
                token_type_ids = batch_tensors['token_type_ids'].to(device)
                mlm_labels = mlm_labels.to(device)
                points = points.to(device)
                mlm_loss, geo_loss, preds = model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    mlm_labels,
                    points,
                    True
                )
                dev_mlm_losses.append(mlm_loss.item())
                if args.mtl:
                    dev_true.extend(points.detach().cpu().tolist())
                    dev_preds.extend(preds)

            test_mlm_losses = list()
            if args.mtl:
                test_true = list()
                test_preds = list()
            for batch_tensors, mlm_labels, points in test_loader:
                input_ids = batch_tensors['input_ids'].to(device)
                attention_mask = batch_tensors['attention_mask'].to(device)
                token_type_ids = batch_tensors['token_type_ids'].to(device)
                mlm_labels = mlm_labels.to(device)
                points = points.to(device)
                mlm_loss, geo_loss, preds = model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    mlm_labels,
                    points,
                    True
                )
                test_mlm_losses.append(mlm_loss.item())
                if args.mtl:
                    test_true.extend(points.detach().cpu().tolist())
                    test_preds.extend(preds)

        # Print and store results
        ppl_dev, ppl_test = np.exp(np.mean(dev_mlm_losses)), np.exp(np.mean(test_mlm_losses))
        if args.mtl:
            dist_dev = median_dist(dev_preds, dev_true, train_dataset.scaler, 'points')
            dist_test = median_dist(test_preds, test_true, train_dataset.scaler, 'points')
            print(ppl_dev, ppl_test, dist_dev, dist_test, args.lr, epoch)
            with open('results/{}.txt'.format(filename), 'a+') as f:
                f.write('{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.0e}\t{}\n'.format(
                    ppl_dev,
                    ppl_test,
                    dist_dev,
                    dist_test,
                    args.lr,
                    epoch
                ))
        else:
            print(ppl_dev, ppl_test, args.lr, epoch)
            with open('results/{}.txt'.format(filename), 'a+') as f:
                f.write('{:.3f}\t{:.3f}\t{:.0e}\t{}\n'.format(
                    ppl_dev,
                    ppl_test,
                    args.lr,
                    epoch
                ))

        if args.lr == 3e-05:
            torch.save(model.state_dict(), 'trained/{}_{:02d}.torch'.format(filename, epoch))


if __name__ == '__main__':
    main()
