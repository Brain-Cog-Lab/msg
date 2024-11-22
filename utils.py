import pandas as pd
from pandas import DataFrame
import os
import argparse
import logging
import logging.handlers
import torch
import numpy as np
import random

## eval acc
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


## seed
def seed_all(seed=1000):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

## Logging
class FormatterNoInfo(logging.Formatter):
    def __init__(self, fmt='%(levelname)s: %(message)s'):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        if record.levelno == logging.INFO:
            return str(record.getMessage())
        return logging.Formatter.format(self, record)


def setup_default_logging(default_level=logging.INFO, log_path=None):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FormatterNoInfo())
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)
    if log_path:
        file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=(1024 ** 2 * 2), backupCount=3)
        file_formatter = logging.Formatter("%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s")
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)

## excel
def args2dataframe(data):
    if isinstance(data, argparse.Namespace):
        data = vars(data)
    if isinstance(data, dict):
        for key in data:
            data[key] = [data[key]]
        df = DataFrame(data)
        return df

    if not isinstance(data, (argparse.Namespace, dict)):
        raise ValueError("data must be a dictionary or argparse.Namespace")


def result2excel(file_path, args):
    df = args2dataframe(args)
    if os.path.exists(file_path):
        pd.set_option('display.notebook_repr_html', False)
        old_df = pd.read_excel(io=file_path)
        df = pd.concat([old_df, df], axis=0, join='outer')
    df.to_excel(file_path, index=False)


def result2csv(file_path, args):
    df = args2dataframe(args)
    if os.path.exists(file_path):
        pd.set_option('display.notebook_repr_html', False)
        old_df = pd.read_csv(file_path)
        df = pd.concat([old_df, df], axis=0, join='outer')
    df.to_csv(file_path, index=False)