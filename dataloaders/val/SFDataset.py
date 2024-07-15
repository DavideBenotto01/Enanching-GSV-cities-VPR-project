from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

DATASET_ROOT = 'MLDL_datasets/sf_xs/'
GT_ROOT = 'datasets/'

path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception(f'Please make sure the path {DATASET_ROOT} to sf_xs dataset is correct')

if not path_obj.joinpath('val', 'database') or not path_obj.joinpath('val', 'queries'):
    raise Exception(f'Please make sure the directories query and ref are situated in the directory {DATASET_ROOT}')


class SFDataset(Dataset):
    def __init__(self, input_transform=None, val_test='val'):
        self.input_transform = input_transform
        self.val_test = val_test
        # reference images names
        self.dbImages = np.load(GT_ROOT + f'sf_xs/sfxs_{self.val_test}_db.npy')  # TODO

        # query images names
        self.qImages = np.load(GT_ROOT + f'sf_xs/sfxs_{self.val_test}_qry.npy')  # TODO

        # ground truth
        self.ground_truth = np.load(GT_ROOT + f'sf_xs/sfxs_{self.val_test}_gt.npy', allow_pickle=True)

        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))

        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT + self.val_test + "/" + self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)
