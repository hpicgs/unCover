# This is just for downloading the generator.

import os
import argparse
import requests


def download_grover(model_type, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for ext in ['data-00000-of-00001', 'index', 'meta']:
        r = requests.get(f'https://storage.googleapis.com/grover-models/{model_type}/model.ckpt.{ext}', stream=True)
        with open(os.path.join(model_dir, f'model.ckpt.{ext}'), 'wb') as f:
            file_size = int(r.headers["content-length"])
            if file_size < 1000:
                raise ValueError("File not available?")
            chunk_size = 1000
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
        print(f"Just downloaded {model_type}/model.ckpt.{ext}!", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download a model!')
    parser.add_argument(
        'model_type',
        type=str,
        help='Valid model names: (base|large|mega)',
    )
    parser.add_argument(
        'model_dir',
        type=str,
        help='Please provide the full path to ModelDirectory',
    )
    model_type = parser.parse_args().model_type
    model_dir = os.path.join(parser.parse_args().model_dir, "grover-" + model_type)
    download_grover(model_type, model_dir)
