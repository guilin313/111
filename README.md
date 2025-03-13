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




from cProfile import label

import numpy as np
import tifffile as tiff
import torch
from IPython.core.pylabtools import figsize
from numpy.ma.core import indices
from scipy.constants import precision
from skimage.feature import peak_local_max
from scipy.spatial import cKDTree
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from torch import cdist
import itertools

file_path = "/home/guilin/PycharmProjects/project226/result_tif/pred/heatmap_pred_0.tif"
file_path_true = "/home/guilin/PycharmProjects/project226/result_tif/true/heatmap_true_0.tif"
image = tiff.imread(file_path)
image_true = tiff.imread(file_path_true)
print("形状：", image.shape)

def match_coords(image):
    channel_1 = image[0]
    channel_2 = image[1]

    min_distance = 5
    threshold_abs = 0.5
    max_distance = 20

    pre_coords = peak_local_max(channel_1, min_distance=min_distance, threshold_abs = threshold_abs)
    post_coords = peak_local_max(channel_2, min_distance=min_distance, threshold_abs=threshold_abs)

    tree = cKDTree(post_coords)
    #distances, indices = tree.query(pre_coords, k=7)
    indices = tree.query_ball_point(pre_coords, max_distance)

    # valid_matches = distances < max_distance
    #
    # matched_pre_coords = pre_coords[valid_matches]
    # matched_post_coords = post_coords[indices[valid_matches]]
    matched_pre_coords = []
    matched_post_coords = []

    # for i,(dists, idxs) in enumerate(zip(distances, indices)):
    #     valid_idx = idxs[dists < max_distance]
    #
    #     if len(valid_idx)>0:
    #         matched_pre_coords.append(pre_coords[i])
    #         matched_post_coords.append(post_coords[valid_idx])

    for i, matched_idx in enumerate(indices):
        if matched_idx:
            matched_pre_coords.append(pre_coords[i])
            matched_post_coords.append(post_coords[matched_idx])

    matched_pre_coords = np.array(matched_pre_coords, dtype=int)
    matched_post_coords = [np.array(p, dtype = int) for p in matched_post_coords]

    for i in range(min(10, len(matched_pre_coords))):
        print(f"前突触{i}坐标:{matched_pre_coords[i]}")
        print(f"匹配的后突触坐标：{matched_post_coords[i]}\n")

    matched_post_coords = [coord for sublist in matched_post_coords for coord in sublist]
    matched_post_coords = np.array(matched_post_coords, dtype=int)

    return matched_pre_coords,matched_post_coords,channel_1,channel_2

def comppute_f1_score(pred_pre, pred_post, true_pre, true_post, pre_radius=50, post_radius=50):
    pred_pre_tensor = torch.tensor(pred_pre, dtype=torch.float32)
    pred_post_tensor = torch.tensor(pred_post, dtype=torch.float32)
    true_pre_tensor = torch.tensor(true_pre, dtype=torch.float32)
    true_post_tensor = torch.tensor(true_post, dtype=torch.float32)

    pre_dist_matrix = cdist(pred_pre_tensor, true_pre_tensor)
    post_dist_matrix = cdist(pred_post_tensor, true_post_tensor)

    pre_match = pre_dist_matrix <= pre_radius
    post_match = post_dist_matrix <= post_radius

    tp_count = 0
    matched_true_indices_pre = set()
    matched_true_indices_post = set()

    # for pred_idx in range(len(pred_pre)):
    #     true_indices_pre = np.where(pre_match[pred_idx])[0]
    #     true_indices_post = np.where(post_match[pred_idx])[0]
    #
    #     for true_idx in true_indices_pre:
    #         if true_idx in true_indices_post and true_idx not in matched_true_indices:
    #             tp_count+=1
    #             matched_true_indices.add(true_idx)
    #             break

    for pred_idx in range(len(pred_pre)):
        true_indices_pre = np.where(pre_match[pred_idx])[0]

        for true_idx in range(len(true_indices_pre)):
            if true_idx not in matched_true_indices_pre:
                tp_count+=1
                matched_true_indices_pre.add(true_idx)
                break

    for pred_idx_post in range(len(pred_post)):
        true_indices_post = np.where(post_match[pred_idx_post])[0]

        for true_idx_1 in range(len(true_indices_post)):
            if true_idx_1 not in matched_true_indices_post:
                tp_count+=1
                matched_true_indices_post.add(true_idx_1)
                break

    tp = tp_count
    fp = len(pred_pre) + len(pred_post) - tp
    fn = len(true_pre) + len(true_post) - tp

    precision = tp / (tp + fp) if (tp + fp) >0 else 0
    recall = tp / (tp + fn) if (fp + fn) >0 else 0
    f1 = (2*precision*recall) / (precision + recall) if (precision + recall)>0 else 0

    return f1,precision,recall,tp,fp,fn

matched_pre_coords, matched_post_coords, channel_1, channel_2 = match_coords(image)
matched_pre_coords_true, matched_post_coords_true ,channel_1_true, channel_2_true = match_coords(image_true)

# print("预测图匹配的前突触坐标（z,y,x)")
# print(matched_pre_coords[:10])
#
# print("预测图匹配的后突触坐标（z，y，x）")
# print(matched_post_coords[:10])
#
# print("\n真实图匹配的前突触坐标（z,y,x)")
# print(matched_pre_coords_true[:10])
#
# print("真实图匹配的后突触坐标（z，y，x）")
# print( matched_post_coords_true[:10])

print("match_pre_coords.shape:", matched_pre_coords.shape)
print("match_pre_coords[:, 2].shape:", matched_pre_coords[:, 2].shape)
print("match_pre_coords[:, 1].shape:", matched_pre_coords[:, 1].shape)
print("match_post_coords.shape:", matched_post_coords.shape)
print("match_post_coords[:, 2].shape:", matched_post_coords[:, 2].shape)
print("match_post_coords[:, 1].shape:", matched_post_coords[:, 1].shape)

f1, precision, recall, tp, fp, fn = comppute_f1_score(matched_pre_coords, matched_post_coords,matched_pre_coords_true, matched_post_coords_true)

print(f"precision:{precision:.3f}")
print(f"recall:{recall:.3f}")
print(f"F1 score:{f1:3f}")
print(f"tp:{tp},fp:{fp},fn:{fn}")
# z_index = matched_pre_coords[0][0]
#
# plt.imshow(channel_1[z_index], cmap="Reds", alpha=0.6)
# plt.imshow(channel_2[z_index], cmap="Blues", alpha=0.6)
#
# plt.scatter(matched_pre_coords[:, 2], matched_pre_coords[:, 1], color="red", label="Pre-Synapse")
# plt.scatter(matched_post_coords[:, 2],matched_post_coords[:, 1], color="blue", label="Post-Synapse")
#
# plt.legend()
# plt.title(f"Matched Synapses on Slice {z_index}")
# plt.show()
unique_z_slices_pre = np.unique(matched_pre_coords[:, 0])
unique_z_slices_post = np.unique(matched_post_coords[:, 0])

# for z in unique_z_slices:
#     plt.figure(figsize=(6, 6))
#
#     slice_image = channel_1[z]
#
#     points_in_slice = matched_pre_coords[matched_pre_coords[:, 0] == z]
#
#     plt.imshow(slice_image, cmap="gray")
#
#     plt.scatter(points_in_slice[:, 2], points_in_slice[: ,1], color="red", label=f"z={z}")
#
#     plt.title(f"Synapse Slice Z={z}")
#     plt.legend()
#     plt.axis("off")
#     plt.show()

num = max(len(unique_z_slices_pre), len(unique_z_slices_post))

fig, axes = plt.subplots(2, num, figsize=(num *4,8))

for i,z in enumerate(unique_z_slices_pre):
    slice_pre =  channel_1[z]
    pre_points_in_slice = matched_pre_coords[matched_pre_coords[:, 0] == z]

    axes[0, i].imshow(slice_pre, cmap="gray")
    axes[0, i].scatter(pre_points_in_slice[:, 2], pre_points_in_slice[:, 1], color="red", label=f"Z={z}")
    axes[0, i].set_title(f"Rre-Synapse (Z={z})")
    axes[0, i].axis("off")

for i, z in enumerate(unique_z_slices_post):
    slice_post = channel_2[z]
    post_points_in_slice = matched_post_coords[matched_post_coords[:, 0] == z]

    axes[1, i].imshow(slice_post, cmap="gray")
    axes[1, i].scatter(post_points_in_slice[:, 2], post_points_in_slice[:, 1], color="blue", label=f"Z={z}")
    axes[1, i].set_title(f"Rost-Synapse (Z={z})({post_points_in_slice[:, 2]},{post_points_in_slice[:, 1]})")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()

