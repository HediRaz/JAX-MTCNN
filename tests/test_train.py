from train.utils_dataset import PNetDataset, create_dataloaders
import numpy as np
from tqdm import tqdm


def test_pnet_dataset():
    ds = PNetDataset()
    train_dl, test_dl = create_dataloaders(ds, batch_size=8)
    all_fc = np.array([0, 0])

    for batch in tqdm(train_dl):
        for img, fc, bbx in zip(batch["img"], batch["fc"], batch["bbx"]):
            all_fc += fc
    print(all_fc)
