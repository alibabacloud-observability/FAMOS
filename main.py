import argparse
import os.path
import time
import yaml
import json
import torch
from torch_geometric.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from model.base_model import base_model
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils import set_seed, save_result, output_metrics, get_dataset, print_label


def get_mutimodal_feature(batch_data):
    node_length = [len(list_x) for list_x in batch_data.x]
    batch_trace_feature = torch.stack([x['trace_feature'] for list_x in batch_data.x for x in list_x])
    batch_metrics_feature = torch.stack(
        [x['metrics_feature'][:39, ] for list_x in batch_data.x for x in list_x]).permute(0, 2, 1)
    # Logs are variable-length time series and need padding
    batch_logs_feature = [x['logs_feature'] for list_x in batch_data.x for x in list_x]
    sequence_lengths = [len(seq) for seq in batch_logs_feature]
    # Sort by length in descending order and save the indices to restore the original order
    sequence_lengths, perm_idx = torch.sort(torch.LongTensor(sequence_lengths), descending=True)
    batch_logs_feature = [batch_logs_feature[i] for i in perm_idx]
    # Perform padding
    max_length = max(sequence_lengths)
    padded_sequences = torch.zeros(len(batch_logs_feature), max_length, batch_logs_feature[0].shape[1])
    for i, seq in enumerate(batch_logs_feature):
        end = sequence_lengths[i]
        padded_sequences[i, :end, :] = seq
    # Pack the sequences
    log_packed_input = pack_padded_sequence(padded_sequences, sequence_lengths, batch_first=True)
    return batch_trace_feature.cuda(), log_packed_input.cuda(), batch_metrics_feature.cuda(), perm_idx, batch_data.edge_index.cuda(), node_length


def testing(model):
    # Testing
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_data in test_loader:
            batch_trace_feature, log_packed_input, batch_metrics_feature, perm_idx, edge_index, node_length = get_mutimodal_feature(
                batch_data)
            label = (batch_data.label).cuda()
            output = model(batch_trace_feature, log_packed_input, batch_metrics_feature, perm_idx, edge_index,
                           node_length)
            y_pred += list(torch.argsort(output, dim=1, descending=True).cpu().numpy())
            y_true += list(label.cpu().numpy())
    result_df = pd.DataFrame(
        {
            "y_pred": [list(item) for item in y_pred],
            "y_true": y_true
        }
    )
    return output_metrics(result_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAMOS")
    parser.add_argument("--config", type=str, default="train-ticket.yaml")
    args = parser.parse_args()
    path = "configs/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    print(json.dumps(config, indent=4))
    if not os.path.exists('./result'): os.makedirs('./result')
    if not os.path.exists('./dataset'): os.makedirs('./dataset')
    set_seed(config['seed'])
    model = base_model(embedding_size=config['word_embedding_size'], metric_feature_size=config['metric_feature_size'],
                       num_classes=config['num_classes'])
    model = model.cuda()
    # Create DataLoader
    X_train, X_test = get_dataset(config)
    y_train = [data.label for data in X_train]
    y_test = [data.label for data in X_test]
    print_label(y_train, 'Training set')
    print_label(y_test, 'Testing set')
    train_loader = DataLoader(X_train, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(X_test, batch_size=config['batch_size'], shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config['lr']), weight_decay=5e-6)
    average_loss_dict = {}
    iter = tqdm(range(config['epoch']), desc="Training: ")
    for epoch in iter:
        model.train()
        total_loss = 0.0
        for batch_data in train_loader:
            batch_trace_feature, log_packed_input, batch_metrics_feature, perm_idx, edge_index, node_length = get_mutimodal_feature(
                batch_data)
            label = (batch_data.label).cuda()
            output = model(batch_trace_feature, log_packed_input, batch_metrics_feature, perm_idx, edge_index,
                           node_length)
            loss = criterion(output, label)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        average_loss = total_loss / len(train_loader)
        average_loss_dict[epoch] = average_loss
        iter.set_postfix({'avg_loss': average_loss})
    model.eval()
    y_pred = []
    y_true = []
    s_t = time.time()
    with torch.no_grad():
        for batch_data in test_loader:
            batch_trace_feature, log_packed_input, batch_metrics_feature, perm_idx, edge_index, node_length = get_mutimodal_feature(
                batch_data)
            label = (batch_data.label).cuda()
            output = model(batch_trace_feature, log_packed_input, batch_metrics_feature, perm_idx, edge_index,
                           node_length)
            y_pred += list(torch.argsort(output, dim=1, descending=True).cpu().numpy())
            y_true += list(label.cpu().numpy())
    e_t = time.time()
    i_pre_time = (e_t - s_t) / len(X_test)
    print(f'Inference time: {i_pre_time}')
    result_df = pd.DataFrame(
        {
            "y_pred": [list(item) for item in y_pred],
            "y_true": y_true
        }
    )
    save_result(result_df, config)
