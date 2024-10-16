import argparse
from pathlib import Path


def get_arguments(additional_args=[]):
    parser = argparse.ArgumentParser(description='Training SFC')

    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--fit-model', type=bool, default=True, help='Fit model or load model')

    parser.add_argument('--show-loss-function', type=bool, default=False, help='Show loss function plot after training')
    parser.add_argument('--save-addr-prefix', type=str, default='artifacts/weights/sfc')
    parser.add_argument('--load-weight-addr', type=str, default='weights/sfc/best/epoch_396_train_acc_0.93.h5')
    parser.add_argument('--mocap-data-dir', type=str, default='mocap-data')
    
    
    for parser_arg, parser_kwargs in additional_args:
        parser.add_argument(parser_arg, **parser_kwargs)

    args = parser.parse_args()
    return args
