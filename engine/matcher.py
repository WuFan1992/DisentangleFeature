import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from methods.EmbPose.varkpnetmodel import *


# -----------------------------
# 1. 网络定义（和训练时一致）
# -----------------------------
# 注意：这里使用你训练好的网络类 VUDNet
# 假设你已经有 VUDNet 类和所有子模块的代码
def pad_to_same_height(img1, img2):
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    
    max_h = max(h1, h2)
    
    def pad(img, target_h):
        h, w, c = img.shape
        pad_h = target_h - h
        return np.pad(img, ((0,pad_h),(0,0),(0,0)), mode='constant')
    
    return pad(img1, max_h), pad(img2, max_h)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化网络
net= VUDNet(feature_dim=64, dim_geo=32,
                 dim_app=16)
net = net.to(device)

# 加载训练好的权重
checkpoint_path = "checkpoints/kpnet_pcorrect_01.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
net.load_state_dict(checkpoint)  # 假设你保存的是 model_state_dict
net.eval()

# -----------------------------
# 2. 图像读入 + 推理
# -----------------------------
def load_image(path, device):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.permute(2,0,1).unsqueeze(0).to(device)
    return img_tensor, img

img_path1 = "datasets/MegaDepth_v1/0022/dense0/imgs/186069410_b743faece0_o.jpg"
img_path2 = "datasets/MegaDepth_v1/0022/dense0/imgs/511190120_77bee89b37_o.jpg"

#img_path1 = "datasets/MegaDepth_v1/0022/dense0/imgs/186069410_b743faece0_o.jpg"
#img_path2 = "datasets/MegaDepth_v1/0022/dense0/imgs/307037213_48891bca3e_o.jpg"

#img_path1 = "datasets/MegaDepth_v1/0022/dense0/imgs/frame-000413.color.png"
#img_path2 = "datasets/MegaDepth_v1/0022/dense0/imgs/frame-000240.color.png"

#img_path1 = "datasets/MegaDepth_v1/0022/dense0/testimgs/republique1.jpg"
#img_path2 = "datasets/MegaDepth_v1/0022/dense0/testimgs/republique2.jpg"

img_tensor1, img1 = load_image(img_path1, device)
img_tensor2, img2 = load_image(img_path2, device)


with torch.no_grad():
    out1 = net(img_tensor1)
    out2 = net(img_tensor2)

# -----------------------------
# 3. 提取关键点（heatmap*reliability*(1-sigma)）
# -----------------------------
def extract_keypoints(heatmap, reliability, sigma, img_shape, num_keypoints=200):
    hmap = heatmap.squeeze().cpu().numpy()
    rel = reliability.squeeze().cpu().numpy()
    sig = sigma.squeeze().cpu().numpy()
    
    # ===== 📊 统计信息 =====
    print("\n" + "="*60)
    print("📊 关键点提取的参数统计")
    print("="*60)
    print(f"Heatmap  - min: {hmap.min():.4f}, max: {hmap.max():.4f}, mean: {hmap.mean():.4f}")
    print(f"Reliability - min: {rel.min():.4f}, max: {rel.max():.4f}, mean: {rel.mean():.4f}")
    print(f"Sigma    - min: {sig.min():.4f}, max: {sig.max():.4f}, mean: {sig.mean():.4f}")
    
    score = hmap * rel * (1 - sig)
    print(f"Score (hmap*rel*(1-sig)) - min: {score.min():.4f}, max: {score.max():.4f}, mean: {score.mean():.4f}")
    
    score_without_rel = hmap * (1 - sig)
    print(f"Score (hmap*(1-sig))     - min: {score_without_rel.min():.4f}, max: {score_without_rel.max():.4f}, mean: {score_without_rel.mean():.4f}")
    print("="*60 + "\n")

    # ✅ ====== 新增：mask 掉边界 ======
    margin = 16   # 👉 可以调：8 / 16 / 32
    score[:margin, :] = 0
    score[-margin:, :] = 0
    score[:, :margin] = 0
    score[:, -margin:] = 0

    # 提取候选点
    coords = np.argwhere(score > 0.0)[:, [1, 0]]  # (x,y)
    scores = score[score > 0.0]

    # 排序
    idx = np.argsort(scores)[::-1]
    coords = coords[idx]
    scores = scores[idx]

    # score阈值过滤
    mask = scores > 0.001
    coords = coords[mask]
    scores = scores[mask]
    print(f"{len(coords)}")

    # top-K
    coords = coords[:num_keypoints]
    scores = scores[:num_keypoints]

    # 映射回原图
    Hf, Wf = hmap.shape
    Hi, Wi = img_shape[:2]
    
    scale_x = Wi / Wf
    scale_y = Hi / Hf
    
    coords = coords.astype(np.float32)
    coords[:, 0] *= scale_x
    coords[:, 1] *= scale_y
    
    return coords, scores

coords1, scores1 = extract_keypoints(
    out1['heatmap'], out1['reliability'], out1['sigma'], img1.shape
)
coords2, scores2 = extract_keypoints(
    out2['heatmap'], out2['reliability'], out2['sigma'], img2.shape
)

# -----------------------------
# 4. 匹配关键点 (余弦相似度 + MNN + 距离过滤)
# -----------------------------
def coords_to_feat(feat_map, coords, img_shape):
    # feat_map: [1,C,H,W], coords: [N,2] 原图坐标
    B,C,H,W = feat_map.shape
    x = np.clip((coords[:,0] * W / img_shape[1]).astype(int), 0, W-1)
    y = np.clip((coords[:,1] * H / img_shape[0]).astype(int), 0, H-1)
    idx = y * W + x
    feat = feat_map.squeeze(0).permute(1,2,0).reshape(-1,C)[idx]
    return F.normalize(feat, dim=1).cpu().numpy()

f1 = coords_to_feat(out1['f_inv'], coords1, img1.shape)
f2 = coords_to_feat(out2['f_inv'], coords2, img2.shape)

# -----------------------------
# 4. 匹配关键点 (余弦相似度 + MNN + 距离过滤)
# -----------------------------
sim = f1 @ f2.T

# 正向最近邻
idx12 = np.argmax(sim, axis=1)
# 反向最近邻
idx21 = np.argmax(sim, axis=0)

# mutual nearest neighbor
matches = []
for i, j in enumerate(idx12):
    if idx21[j] == i:
        """
        # 添加距离过滤：计算像素距离
        pt1 = coords1[i]
        pt2 = coords2[j]
        dist = np.linalg.norm(pt1 - pt2)
        if dist < 100:  # 像素距离阈值，可调
        """
        matches.append([i, j])

matches = np.array(matches)

print(f"Total MNN matches: {len(matches)} (after distance filter)")

# -----------------------------
# 5. 可视化
# -----------------------------
def visualize(img1, img2, coords1, coords2, matches, heatmap1, heatmap2, sigma1, sigma2):
    
    # ✅ ====== 新增：padding ======
    img1, img2 = pad_to_same_height(img1, img2)

    concat_img = np.concatenate([img1, img2], axis=1)
    
    plt.figure(figsize=(15,7))
    plt.imshow(concat_img)
    
    for m in matches:
        pt1 = coords1[m[0]]
        pt2 = coords2[m[1]] + np.array([img1.shape[1],0])
        
        plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],'y',linewidth=1)
    
    plt.title("Keypoint Matches")
    plt.axis('off')
    plt.show()
    
    # Heatmap + Variance（不变）
    fig, axes = plt.subplots(2,2,figsize=(12,10))
    axes[0,0].imshow(heatmap1.squeeze().cpu(), cmap='hot')
    axes[0,0].set_title("Heatmap img1")
    axes[0,0].axis('off')
    
    axes[0,1].imshow(heatmap2.squeeze().cpu(), cmap='hot')
    axes[0,1].set_title("Heatmap img2")
    axes[0,1].axis('off')
    
    axes[1,0].imshow(sigma1.squeeze().cpu(), cmap='viridis')
    axes[1,0].set_title("Variance img1")
    axes[1,0].axis('off')
    
    axes[1,1].imshow(sigma2.squeeze().cpu(), cmap='viridis')
    axes[1,1].set_title("Variance img2")
    axes[1,1].axis('off')
    
    plt.show()

visualize(img1, img2, coords1, coords2, matches, out1['heatmap'], out2['heatmap'], out1['sigma'], out2['sigma'])