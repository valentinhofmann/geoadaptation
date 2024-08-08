import argparse
import pickle
import random

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from helpers_id import *
from model_id import *


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
    parser.add_argument('--domain', default="twitter", type=str, help='Type of data.')
    parser.add_argument('--device', default=None, type=int, required=True, help='Selected CUDA device.')
    args = parser.parse_args()

    # Load data
    if args.domain == "news":
        train = pd.read_json('../data/{0}/train_id_news.json'.format(args.data), lines=True)
        language_id_mappings = {l_id: i for i, l_id in enumerate(set(train.id))}
        train["id"] = train.id.apply(lambda x: language_id_mappings[x])
        train_dataset = DatasetForID(train)
        dev = pd.read_json('../data/{0}/dev_id_news.json'.format(args.data), lines=True)
        dev["id"] = dev.id.apply(lambda x: language_id_mappings[x])
        dev_dataset = DatasetForID(dev)
        test = pd.read_json('../data/{0}/test_id_news.json'.format(args.data), lines=True)
        test["id"] = test.id.apply(lambda x: language_id_mappings[x])
        test_dataset = DatasetForID(test)
    else:
        train = pd.read_json('../data/{0}/train_id.json'.format(args.data), lines=True)
        language_id_mappings = {l_id: i for i, l_id in enumerate(set(train.id))}
        train["id"] = train.id.apply(lambda x: language_id_mappings[x])
        train_dataset = DatasetForID(train)
        dev = pd.read_json('../data/{0}/dev_id.json'.format(args.data), lines=True)
        dev["id"] = dev.id.apply(lambda x: language_id_mappings[x])
        dev_dataset = DatasetForID(dev)
        test = pd.read_json('../data/{0}/test_id.json'.format(args.data), lines=True)
        test["id"] = test.id.apply(lambda x: language_id_mappings[x])
        test_dataset = DatasetForID(test)

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
    if args.domain == "news":
        print('Using news data...')

    # Define filename
    filename = args.data
    if args.mtl:
        filename += '_{}'.format(args.mtl)
    if args.head:
        filename += '_{}'.format(args.head)
    filename += '_{}'.format(args.generation)
    filename_out = filename
    if args.domain == "news":
        filename_out += "_news"

    best_accuracy, *_ = get_best('results_id/{}.txt'.format(filename_out))
    print('Best accuracy so far: {}'.format(best_accuracy))

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, model_max_length=128)

    # Load data loaders
    cltr = CollatorForID(tok)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=cltr, shuffle=True, drop_last=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=cltr)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=cltr)

    # Define dim of output layer
    output_dim = train_dataset.n_labels

    # Define device
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    # Initialize model
    if args.model == 'classla/bcms-bertic':
        model = ElectraForID(args.model, output_dim)
        print('Loading model weights from {}...'.format(filename))
        pretrained_dict = torch.load('trained/{}.torch'.format(filename), map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'electra' in k}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    elif args.model == 'vesteinn/ScandiBERT':
        model = RobertaForID(args.model, output_dim)
        print('Loading model weights from {}...'.format(filename))
        pretrained_dict = torch.load('trained/{}.torch'.format(filename), map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'roberta' in k}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        model = BertForID(args.model, output_dim)
        print('Loading model weights from {}...'.format(filename))
        pretrained_dict = torch.load('trained/{}.torch'.format(filename), map_location=device)
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
        for i, (batch_tensors, labels) in enumerate(train_loader):
            input_ids = batch_tensors['input_ids'].to(device)
            attention_mask = batch_tensors['attention_mask'].to(device)
            token_type_ids = batch_tensors['token_type_ids'].to(device)
            if i == 0:
                print(input_ids[0, :])
            labels = labels.to(device)
            optimizer.zero_grad()
            loss, preds = model(
                input_ids,
                attention_mask,
                token_type_ids,
                labels
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
            for batch_tensors, labels in dev_loader:
                input_ids = batch_tensors['input_ids'].to(device)
                attention_mask = batch_tensors['attention_mask'].to(device)
                token_type_ids = batch_tensors['token_type_ids'].to(device)
                labels = labels.to(device)
                loss, preds = model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    labels
                )
                dev_true.extend(labels.detach().cpu().tolist())
                dev_preds.extend(preds)

            test_true = list()
            test_preds = list()
            for batch_tensors, labels in test_loader:
                input_ids = batch_tensors['input_ids'].to(device)
                attention_mask = batch_tensors['attention_mask'].to(device)
                token_type_ids = batch_tensors['token_type_ids'].to(device)
                labels = labels.to(device)
                loss, preds = model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    labels
                )
                test_true.extend(labels.detach().cpu().tolist())
                test_preds.extend(preds)

        # Print and store results
        accuracy_dev = accuracy_score(dev_true, dev_preds)
        accuracy_test = accuracy_score(test_true, test_preds)
        print(accuracy_dev, accuracy_test, args.lr, epoch)
        with open('results_id/{}.txt'.format(filename_out), 'a+') as f:
            f.write('{:.3f}\t{:.3f}\t{:.0e}\t{}\n'.format(
                accuracy_dev,
                accuracy_test,
                args.lr,
                epoch
            ))

        # Store predictions
        if best_accuracy is None or accuracy_dev > best_accuracy:
            best_accuracy = accuracy_dev
            with open('preds_id/{}.p'.format(filename_out), 'wb') as f:
                pickle.dump([dev_preds, dev_true, test_preds, test_true], f)


if __name__ == '__main__':
    main()
