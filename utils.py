import numpy as np
import random
import torch
import sys
import os
import pytz
from datetime import datetime
from diskcache import Cache
from collections import Counter
from tqdm import tqdm
import pickle
import urllib.request
import zipfile

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def output_metrics(result_df):
    from sklearn.metrics import precision_score, recall_score, f1_score
    try:
        y_pred = np.array([np.array(eval(item)) for item in list(result_df['y_pred'])])
    except:
        y_pred = np.array([np.array((item)) for item in list(result_df['y_pred'])])
    y_true = np.array(list(result_df['y_true']))
    y_pred = np.array([item[0] for item in y_pred])
    # Macro Average
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(
        f"Macro Precision: {macro_precision * 100:.2f}% Recall: {macro_recall * 100:.2f}% F1-Score: {macro_f1 * 100:.2f}%")


def save_result(result_df, config):
    save_result_dir = config['result_dir']
    script_name = os.path.basename(sys.argv[0])
    beijing_tz = pytz.timezone('Asia/Shanghai')
    beijing_time = datetime.now(beijing_tz)
    time_str = beijing_time.strftime('%m-%d_%H-%M')
    file_name = f"result_{os.path.splitext(script_name)[0]}_{time_str}.csv"
    save_result_file = os.path.join(save_result_dir, file_name)
    print(f'save file to {save_result_file}')
    result_df.to_csv(save_result_file, index=False)
    output_metrics(result_df)


def stratified_train_test_split(X, y, test_size=0.25):
    # Get unique labels
    unique_labels = np.unique(y)
    # Initialize train and test sets
    X_train, X_test = [], []
    y_train, y_test = [], []
    for label in unique_labels:
        # Get data corresponding to the label
        X_label = [X[i] for i in range(len(X)) if y[i] == label]
        y_label = [y[i] for i in range(len(X)) if y[i] == label]
        # Calculate split point
        split_point = int(len(X_label) * (1 - test_size))
        # Sequentially split
        X_train.append(X_label[:split_point])
        X_test.append(X_label[split_point:])
        y_train.append(y_label[:split_point])
        y_test.append(y_label[split_point:])
    # Merge data from different labels into final train and test sets
    X_train = [item for a_list in X_train for item in a_list]
    X_test = [item for a_list in X_test for item in a_list]
    y_train = [item for a_list in y_train for item in a_list]
    y_test = [item for a_list in y_test for item in a_list]
    return X_train, X_test, y_train, y_test


def _progress(block_num, block_size, total_size):
    sys.stdout.write('\r>> Downloading %.1f%%' % (
                     float(block_num * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()

def get_data(config):
    dataset_name = config['dataset_name']
    if not os.path.exists(config['dataset_cache_dir']):
        url = config['download_url']
        downloaded_filename =  f'./dataset/{dataset_name}.zip'
        print(f'{dataset_name} not exists')
        urllib.request.urlretrieve(url, downloaded_filename, _progress)
        with zipfile.ZipFile(downloaded_filename, 'r') as zip_ref:
            zip_ref.extractall('./dataset')
    cache = Cache(config['dataset_cache_dir'], size_limit=int(9 * 1e10))
    data_list = []
    chunk_keys = cache.get(config['data_chunk_keys'])
    for key in tqdm(chunk_keys):
        chunk = pickle.loads(cache.get(key))
        data_list.extend(chunk)
    return data_list


def get_dataset(config):
    data_list = get_data(config)
    label_list = [data.label for data in data_list]
    test_size = 3 / (7 + 3)
    X_train, X_test, _, _ = stratified_train_test_split(data_list, label_list, test_size=test_size)
    return X_train, X_test


def print_label(y_label, prefix):
    print(prefix)
    counter = Counter(y_label)
    sorted_counter = sorted(counter.items(), key=lambda x: x[0])
    for element, count in sorted_counter:
        print(f"Class {element}:  {count}")
    print(f"Total :  {len(y_label)}")
