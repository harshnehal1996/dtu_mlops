# -*- coding: utf-8 -*-
import logging
import os
import numpy as np
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# from models.helper import get_project_dir

def main(project_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    suffix = [0,1,2,3,4]
    base_path = os.path.join(project_dir, 'data/raw')
    images = []
    labels = []
    for s in suffix:
        path = os.path.join(base_path, 'train_' + str(s) + '.npz')
        data = np.load(path)
        images.append(data['images'].astype(np.float32))
        labels.append(data['labels'])
    
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)

    with open(os.path.join(project_dir, 'data/processed/train.npz'), 'wb') as outfile:
        np.savez(outfile, images=images, labels=labels)

    path = os.path.join(base_path, 'test.npz')
    data = np.load(path)
    test_images = data['images'].astype(np.float32)
    test_labels = data['labels']

    with open(os.path.join(project_dir, 'data/processed/test.npz'), 'wb') as outfile:
        np.savez(outfile, images=test_images, labels=test_labels)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(project_dir)
