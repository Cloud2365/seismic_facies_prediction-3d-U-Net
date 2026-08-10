import h5py
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
TRAIN_DIR = "./temp_splits/train.h5"
NUM_CLASSES = 10
f = h5py.File(TRAIN_DIR, 'r')
print(f)
print(list(f.keys()))
label = f["label"][:]
print(label.shape)
print("label_type =",type(label))
print("Минимальное значение:",np.min(label))
print("Максимальное значение:",np.max(label))
Z, Y, X = label.shape
print("Размеры:", Z,Y,X)
class_projections_z = np.zeros((NUM_CLASSES,Y,X),dtype=np.float32)
class_projections_x = np.zeros((NUM_CLASSES,Z,Y),dtype=np.float32)
class_projections_y = np.zeros((NUM_CLASSES,Z,X),dtype=np.float32)
overall_projections_z = np.zeros((Y,X),dtype = np.float32)
overall_projections_x = np.zeros((Z,Y),dtype = np.float32)
overall_projections_y = np.zeros((Z,X),dtype = np.float32)

for z in tqdm(range(Z)):
    slice_2d = label[z,:,:]
    mask_overall = (slice_2d>0).astype(np.float32)
    overall_projections_z += mask_overall
    overall_projections_y[z,:] = np.sum(mask_overall,axis=0)
    overall_projections_x[z,:] = np.sum(mask_overall,axis=1)

    for c in range(NUM_CLASSES):
        mask = (slice_2d == c).astype(np.float32)
        class_projections_z[c,:,:] += mask
        class_projections_x[c,z,:] = np.sum(mask,axis=1)
        class_projections_y[c,z,:] = np.sum(mask,axis=0)

overall_projections_z = overall_projections_z/Z
class_projections_z = class_projections_z/Z

overall_projections_x = overall_projections_x/X
class_projections_x = class_projections_x/X

overall_projections_y = overall_projections_y/Y
class_projections_y = class_projections_y/Y



fig, axes = plt.subplots(1,3,figsize=(12,6))
axes = axes.flatten()
im1 = axes[0].imshow(overall_projections_z,cmap="hot",interpolation="nearest")
axes[0].set_title(f"Проекция на ось Z")
axes[0].set_xlabel("X (длинна)")
axes[0].set_ylabel("Y (высота)")
plt.colorbar(im1,ax=axes[0],label="Частота",shrink=0.4,pad=0.04)

im2 = axes[1].imshow(overall_projections_x,cmap="hot",interpolation="nearest")
axes[1].set_title(f"Проекция на ось X")
axes[1].set_xlabel("Y (высота)")
axes[1].set_ylabel("Z (глубина)")
plt.colorbar(im2,ax=axes[1],label="Частота",shrink=0.8,pad=0.04)

im3 = axes[2].imshow(overall_projections_y,cmap="hot",interpolation="nearest")
axes[2].set_title(f"Проекция на ось Y")
axes[2].set_xlabel("X (длинна)")
axes[2].set_ylabel("Z (глубина)")
plt.colorbar(im3,ax=axes[2],label="Частота",shrink=0.6,pad=0.04)


plt.suptitle("Проекции на каждые оси (без фона)")
plt.tight_layout()
plt.savefig("./plots/overall_projections",dpi=150,bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(2,5,figsize=(20,8))
axes = axes.flatten()
for c in range(NUM_CLASSES):
    ax = axes[c]
    im = ax.imshow(class_projections_z[c],cmap ="hot",interpolation="nearest")
    ax.set_title(f"Класс {c}")
    ax.set_xlabel("X (длинна)")
    ax.set_ylabel("Y (высота)")
    plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
plt.suptitle("Проекции для каждого класса на ось Z")
plt.tight_layout()
plt.savefig(f"./plots/class_projections_all_z",dpi=150,bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(2,5,figsize=(20,8))
axes1 = axes.flatten()
for c in range(NUM_CLASSES):
    ax = axes1[c]
    im = ax.imshow(class_projections_x[c],cmap ="hot",interpolation="nearest")
    ax.set_title(f"Класс {c}")
    ax.set_xlabel("Y (высота)")
    ax.set_ylabel("Z (глубина)")
    plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
plt.suptitle("Проекции для каждого класса на ось X")
plt.tight_layout()
plt.savefig(f"./plots/class_projections_all_x",dpi=150,bbox_inches="tight")
plt.show()

fig1, axes1 = plt.subplots(2,5,figsize=(20,8))
axes1 = axes1.flatten()
for c in range(NUM_CLASSES):
    ax = axes1[c]
    im = ax.imshow(class_projections_y[c],cmap ="hot",interpolation="nearest")
    ax.set_title(f"Класс {c}")
    ax.set_xlabel("X (длинна)")
    ax.set_ylabel("Z (глубина)")
    plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
plt.suptitle("Проекции для каждого класса на ось Y")
plt.tight_layout()
plt.savefig(f"./plots/class_projections_all_y",dpi=150,bbox_inches="tight")
plt.show()


print("Проекция на ось z: ")
for c in range(NUM_CLASSES):
    print(f"Класс {c}: средняя частота появления: {np.mean(class_projections_z[c]):.4f}, максимум: {np.max(class_projections_z[c]):.4f}")
