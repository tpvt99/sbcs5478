import numpy as np
from vae.model_torch import AutoEncoder
import math
import pickle
import os.path as osp
import torch
from global_configuration import PROJECT_PATH
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
import os as os

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, max_entries=10000):
        train_data1 = np.load(osp.join(dataset_path, "dataset.npy"))[0: max_entries]
        train_data2 = np.load(osp.join(dataset_path, "badsamples.npy"))[0: max_entries]
        self.dataset = np.concatenate([train_data2, train_data1], axis=0).astype(np.float32)
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        data = self.dataset[index]
        data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.dataset)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

if __name__ == "__main__":
    data_path = osp.join(osp.join(osp.join(PROJECT_PATH, "vae"), "data"), "map3")
    latent_dim = 16
    vae = AutoEncoder(input_shape=(64, 80, 3), latent_dim=latent_dim).to(device)

    result_path = osp.join(osp.join(osp.join(osp.join(PROJECT_PATH, "vae"), "checkpoints"), "map3"), f"latent_{latent_dim}")
    if not osp.exists(result_path):
        os.makedirs(result_path)

    best_loss = math.inf
    epochs = 200
    batch_size = 32

    train_dataset = CustomDataset(data_path, 10000)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, threshold=2e-1, verbose=True, min_lr=1e-5)

    train_size = len(train_dataloader)

    for epoch in range(epochs):
        loss_metric_test = 0
        loss_metric_train = 0
        # Iterate over the batches of the dataset.
        for step, data in enumerate(train_dataloader):
            data = data.to(device)
            reconstructed = vae(data)
            # Compute reconstruction loss
            loss = criterion(data, reconstructed)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_metric_train = loss_metric_train + 1/ (step+1) * (loss.clone().item() - loss_metric_train)

            print(f"Train Epoch {epoch} Iter {step} | Loss {loss_metric_train}")

        if loss_metric_train < best_loss:
            best_loss = loss_metric_train
            torch.save(vae.state_dict(), osp.join(result_path, f"best_model.pt"))
            print(f'Saving best model with loss {best_loss}')

        scheduler.step(loss_metric_train)
