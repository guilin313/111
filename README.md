import torch,gc
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import HDF5KeypointDataset
from model import UNet3D
from tqdm import tqdm
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = HDF5KeypointDataset("/home/guilin/PycharmProjects/project226/data/train")
val_dataset = HDF5KeypointDataset("/home/guilin/PycharmProjects/project226/data/val")

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

model = UNet3D().to(device)

criterion = nn.BCEWithLogitsLoss()
# criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

tb_writer = SummaryWriter()

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0


    with tqdm(train_loader, desc=f"Epoch{epoch+1}/{num_epochs}", unit="batch") as pbar:
        for batch_idx,(volumes, heatmaps) in train_loader:
            volumes, heatmaps = volumes.to(device), heatmaps.to(device)

            optimizer.zero_grad()
            outputs = model(volumes)
            loss = criterion(outputs, heatmaps)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            tb_writer.add_scalar("Loss/Train", loss.item(), epoch * len(train_loader) + batch_idx)

            pbar.set_postfix({"Train Loss": f"{train_loss / (pbar.n + 1):.4f}"})
        torch.cuda.empty_cache()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for volumes, heatmaps in val_loader:
            volumes, heatmaps = volumes.to(device), heatmaps.to(device)
            outputs = model(volumes)
            loss = criterion(outputs, heatmaps)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

    tb_writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

torch.save(model.state_dict(),"unet3d2.pth")
print("训练完成！")

tb_writer.close()
