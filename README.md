![Screenshot from 2025-03-21 11-04-10](https://github.com/user-attachments/assets/9ca2d639-7ed8-4c9a-840b-897def511b5a)
![Screenshot from 2025-03-21 14-02-54](https://github.com/user-attachments/assets/0614289c-5f89-4a4c-9dde-6156a42d40fc)

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



import torch
import torch.nn as nn
import numpy as np
import collections
def get_3d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size, grid_size)

    grid_d = np.arange(grid_size[0], dtype=np.float32)
    grid_h = np.arange(grid_size[1], dtype=np.float32)
    grid_w = np.arange(grid_size[2], dtype=np.float32)

    grid = np.meshgrid(grid_d, grid_w, grid_h, indexing='ij')  # DWH
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([3, -1])

    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 3 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode grid_h
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (D*H*W, D/2)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (D*H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (D*H*W, D/2)

    emb = np.concatenate([emb_d, emb_h, emb_w], axis=1)  # (D*H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

    def visualize_pos_embed_3d(pos_embed, grid_size, embed_dim):
    pos_embed_reshaped = pos_embed.reshape(grid_size[0], grid_size[1], grid_size[2], embed_dim)

    embedding_slice = pos_embed_reshaped[:,:,:,:3]

    fig,axs = plt.subplots(1,3,figsize=(18,6))
    for i,ax in enumerate(axs):
        ax.imshow(embedding_slice[:,:,embedding_slice.shape[2] // 2, i], cmap='viridis')
        ax.set_title(f"embedding channel{i}")
        ax.axis('off')

    plt.show()

class ViTMAEEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = ViTMAEPatchEmbeddings3D(config)
        self.num_patches = self.patch_embeddings.num_patches
        # fixed sin-cos embedding
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.hidden_size), requires_grad=False
        )
        self.patch_size = config.patch_size
        self.config = config
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(
            self.position_embeddings.shape[-1], grid_size=(
                self.config.image_size[0] // self.patch_size[0],
                self.config.image_size[1] // self.patch_size[1],
                self.config.image_size[2] // self.patch_size[2]
            ), add_cls_token=True
        )
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)

    # Copied from transformers.models.vit.modeling_vit.ViTEmbeddings.interpolate_pos_encoding
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, depth: int, height: int, width: int) -> torch.Tensor:

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions :
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_depth = depth // self.patch_size[0]
        new_height = height // self.patch_size[1]
        new_width = width // self.patch_size[2]

        d,w,h = self.patch_embeddings.grid_size
        patch_pos_embed = patch_pos_embed.reshape(1, d, h, w, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 4, 1, 2, 3)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_depth, new_height, new_width),
            mode="trilinear",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def random_masking(self, sequence, noise=None):
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def forward(self, pixel_values, noise=None, interpolate_pos_encoding: bool = False):
        batch_size, num_channels, depth, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        if interpolate_pos_encoding:
            position_embeddings = self.interpolate_pos_encoding(embeddings, depth, height, width)
        else:
            position_embeddings = self.position_embeddings

        # add position embeddings w/o cls token
        embeddings = embeddings + position_embeddings[:, 1:, :]

        # masking: length -> length * config.mask_ratio
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

        # append cls token
        cls_token = self.cls_token + position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore

        class ViTMAEPatchEmbeddings3D(nn.Module):

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size #DHW
        num_channels, hidden_size = config.num_channels, config.hidden_size

        if not isinstance(image_size, collections.abc.Iterable) :
            image_size=(image_size,image_size, image_size)
        if not isinstance(patch_size, collections.abc.Iterable) :
            patch_size=(patch_size,patch_size, patch_size)

        num_patches = (image_size[2] // patch_size[2]) *(image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv3d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values, interpolate_pos_encoding: bool = False):
        batch_size, num_channels, depth, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        x = self.projection(pixel_values) #B hiddensize D H W
        x = x.flatten(2).transpose(1, 2) #B numpatches hiddensize
        return x

import torch

class Config:
    image_size = (32, 32, 32)  # 3D 图像尺寸
    patch_size = (4, 4, 4)  # Patch 大小
    num_channels = 1  # 单通道输入（例如 CT 扫描）
    hidden_size = 128  # Patch 维度

config = Config()
patch_embedding_layer = ViTMAEPatchEmbeddings3D(config)

# 生成一个 batch_size=2 的随机 3D 图像
pixel_values = torch.randn(2, config.num_channels, *config.image_size)  # (2, 1, 32, 32, 32)

patch_embeddings = patch_embedding_layer(pixel_values)
print("Patch Embedding Shape:", patch_embeddings.shape)
# 预期：应该是 (2, num_patches, hidden_size)
# num_patches = (32//4) * (32//4) * (32//4) = 512
# 预期输出：(2, 512, 128)

vit_mae_embeddings = ViTMAEEmbeddings(config)

# 生成一个 batch_size=2 的随机 3D 图像
pixel_values = torch.randn(2, config.num_channels, *config.image_size)  # (2, 1, 32, 32, 32)

# 获取嵌入结果
embeddings, mask, ids_restore = vit_mae_embeddings(pixel_values)

print("Embeddings Shape:", embeddings.shape)
print("Mask Shape:", mask.shape)

import torch
from copy import deepcopy

# 假设 ViTMAEPatchEmbeddings3D, ViTMAEEmbeddings3D, ViTMAEDecoder3D 已经定义
# 以及 ViTMAEConfig3D 也已经存在

# 创建 3D 配置
config = ViTMAEConfig3D(
    image_size=(64, 64, 64),  # 3D 图像尺寸
    patch_size=(16, 16, 16),  # 3D Patch 大小
    num_channels=1,  # 例如医学图像（MRI/CT）通常是单通道
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    decoder_hidden_size=512,
    decoder_num_hidden_layers=8,
    decoder_num_attention_heads=8,
    decoder_intermediate_size=2048
)

# 计算 num_patches
grid_size = (
    config.image_size[0] // config.patch_size[0],  
    config.image_size[1] // config.patch_size[1],  
    config.image_size[2] // config.patch_size[2]  
)
num_patches = grid_size[0] * grid_size[1] * grid_size[2]  # 总 patch 数

# **创建 ViTMAEEmbeddings3D**
embed_layer = ViTMAEEmbeddings3D(config)

# **创建 ViTMAEDecoder3D**
decoder = ViTMAEDecoder(config, num_patches)

# **生成假数据**
batch_size = 2  # 设定 batch size
hidden_states = torch.randn(batch_size, num_patches // 2, config.hidden_size)  # 一半 token 被 mask
ids_restore = torch.randint(0, num_patches, (batch_size, num_patches))  # 伪随机还原索引

# **运行 decoder**
output = decoder(
    hidden_states=hidden_states,
    ids_restore=ids_restore,
    return_dict=True
)

# **测试输出**
print("Decoder 输出 logits 形状:", output.logits.shape)  
# 期望输出: (batch_size, num_patches, patch_size**3 * num_channels)
# (2, num_patches, 16*16*16*1)

print("IDs Restore Shape:", ids_restore.shape)
# 预期：
# embeddings.shape == (batch_size, len_keep+1, hidden_size) (cls token + 变换后patch数)
# mask.shape == (batch_size, num_patches)
# ids_restore.shape == (batch_size, num_patches)




# 假设的测试参数
batch_size = 2
num_patches = 64  # 假设 4x4x4 形状的 Patch grid
hidden_size = 512
decoder_hidden_size = 256

# 创建假 config
config = SimpleNamespace(
    hidden_size=hidden_size,
    decoder_hidden_size=decoder_hidden_size,
    decoder_num_hidden_layers=4,
    decoder_num_attention_heads=8,
    decoder_intermediate_size=1024,
    patch_size=(16, 16, 16),  # 3D Patch 尺寸
    num_channels=1,
    layer_norm_eps=1e-6,
    initializer_range=0.02,
)

# 创建输入张量 (batch_size, num_visible_patches + 1, hidden_size)
hidden_states = torch.randn(batch_size, num_patches // 2 + 1, hidden_size)

# 生成恢复索引 (batch_size, num_patches)
ids_restore = torch.arange(num_patches).repeat(batch_size, 1)

# 创建 ViTMAEDecoder3D 实例
decoder = ViTMAEDecoder3D(config, num_patches)

# 确保模型在评估模式下
decoder.eval()

with torch.no_grad():
    output = decoder(hidden_states, ids_restore)
print("Hidden States Shape:", hidden_states.shape)
print("Ids Restore Shape:", ids_restore.shape)
# 打印输出信息
print("Decoder Output Shape:", output.logits.shape)  # 期望形状: (batch_size, num_patches, patch_size^3 * num_channels)


# 假设的测试参数
batch_size = 2
num_patches = 64  # 假设 4x4x4 形状的 Patch grid
hidden_size = 768
decoder_hidden_size = 768

# 创建假 config
config = ViTMAEConfig(
    image_size=(32, 320, 320),
    hidden_size=hidden_size,
    decoder_hidden_size=decoder_hidden_size,
    decoder_num_hidden_layers=4,
    decoder_num_attention_heads=8,
    decoder_intermediate_size=1024,
    patch_size=(16, 16, 16),  # 3D Patch 尺寸
    num_channels=1,
    layer_norm_eps=1e-6,
    initializer_range=0.02,
)

# 创建输入张量 (batch_size, num_visible_patches + 1, hidden_size)
hidden_states = torch.randn(batch_size, num_patches // 2 + 1, hidden_size)

# 生成恢复索引 (batch_size, num_patches)
ids_restore = torch.arange(num_patches).repeat(batch_size, 1)

# 创建 ViTMAEDecoder3D 实例
decoder = ViTMAEDecoder(config, num_patches)

# 确保模型在评估模式下
decoder.eval()

with torch.no_grad():
    output = decoder(hidden_states, ids_restore)
print("Hidden States Shape:", hidden_states.shape)
print("Ids Restore Shape:", ids_restore.shape)
# 打印输出信息
print("Decoder Output Shape:", output.logits.shape)  # 期望形状: (batch_size, num_patches, patch_size^3 * num_channels)
![Screenshot from 2025-03-20 14-51-46](https://github.com/user-attachments/assets/840a24ea-930f-4798-ba06-5f9702b88581)
![Screenshot from 2025-03-20 14-54-12](https://github.com/user-attachments/assets/20526d78-6942-434b-b4c5-1eab65398517)

