import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
from skimage import transform as sktrans
import logging
from typing import Dict, Any, Union

import matplotlib
import matplotlib.pyplot as plt

try:
    from medical_dl_library.heatmap import GaussianHeatmapGenerator
except ImportError:
    GaussianHeatmapGenerator = None

def get_augmentation_config():
    return {"global_prob": 0.8, "rotate_prob": 0.5, "rotate_limit": 30, "translate_prob": 0.5, "translate_limit": 0.1}

def rotate(angle):
    def func(img):
        ret = []
        for i in range(img.shape[0]):
            rotated = sktrans.rotate(img[i], angle, resize=False, preserve_range=True, mode='constant', cval=0)
            ret.append(rotated)
        return np.array(ret)
    return func

def translate(offsets):
    offsets = tuple(int(o) for o in offsets)
    def func(img):
        ret = []
        for i in range(img.shape[0]):
            old = img[i]
            new = np.zeros_like(old)
            H, W = old.shape
            y_off, x_off = offsets
            if y_off > 0:
                dst_y, src_y = slice(y_off, None), slice(0, H - y_off)
            else:
                dst_y, src_y = slice(0, H + y_off), slice(-y_off, None)
            if x_off > 0:
                dst_x, src_x = slice(x_off, None), slice(0, W - x_off)
            else:
                dst_x, src_x = slice(0, W + x_off), slice(-x_off, None)
            try:
                new[dst_y, dst_x] = old[src_y, src_x]
            except:
                pass
            ret.append(new)
        return np.array(ret)
    return func

def transformer(param_dic):
    fs = []
    if 'rotate' in param_dic: fs.append(rotate(param_dic['rotate']))
    if 'translate' in param_dic: fs.append(translate(param_dic['translate']))
    def trans(*imgs):
        ret = []
        for img in imgs:
            cur_img = img.copy()
            for f in fs: cur_img = f(cur_img)
            ret.append(cur_img.copy())
        return tuple(ret)
    return trans

class DataAugmentor:
    def __init__(self, config=None):
        self.config = config if config else get_augmentation_config()

    def __call__(self, img_np, hm_np):
        if np.random.rand() > self.config['global_prob']: return img_np, hm_np
        params = {}
        H, W = img_np.shape[1], img_np.shape[2]
        if np.random.rand() < self.config['rotate_prob']:
            limit = self.config['rotate_limit']
            params['rotate'] = np.random.uniform(-limit, limit)
        if np.random.rand() < self.config['translate_prob']:
            limit = self.config['translate_limit']
            off_y = int(np.random.uniform(-limit, limit) * H)
            off_x = int(np.random.uniform(-limit, limit) * W)
            params['translate'] = (off_y, off_x)
        if not params: return img_np, hm_np
        trans_func = transformer(params)
        return trans_func(img_np, hm_np)

class CustomLandmarkDataset(Dataset):
    def __init__(self, data_list, heatmap_generator, augment=False, target_size=(512, 512)):
        self.data_list = data_list
        self.heatmap_generator = heatmap_generator
        self.augment = augment
        self.target_size = target_size
        self.augmentor = DataAugmentor(get_augmentation_config())

    def _parse_txt(self, txt_path, w_raw, h_raw):
        coords = []
        try:
            with open(txt_path, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                if len(lines) > 0 and len(lines[0].split()) == 1: lines = lines[1:]
                for line in lines:
                    parts = line.split(',') if ',' in line else line.split()
                    if len(parts) >= 2:
                        x = float(parts[0])
                        y = float(parts[1])
                        if x <= 1.1 and y <= 1.1: x *= w_raw; y *= h_raw
                        coords.append([x, y])
        except Exception:
            pass
        return np.array(coords, dtype=np.float32)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item = self.data_list[index]
        img_path = item['image']
        txt_path = item['label']
        img_pil = Image.open(img_path).convert('L')
        w_raw, h_raw = img_pil.size
        lms = self._parse_txt(txt_path, w_raw, h_raw)

        expected_landmarks = self.heatmap_generator.nb_landmarks
        actual_landmarks = len(lms)
        if actual_landmarks != expected_landmarks:
            raise ValueError(f"Landmark count mismatch in {os.path.basename(txt_path)}: Expected {expected_landmarks}, got {actual_landmarks}.")

        if (w_raw, h_raw) != self.target_size:
            img_pil = img_pil.resize(self.target_size, Image.Resampling.BILINEAR)
            scale_x = self.target_size[0] / w_raw
            scale_y = self.target_size[1] / h_raw
            lms = lms * np.array([scale_x, scale_y])

        img_np = np.array(img_pil)
        img_orig_tensor = torch.from_numpy(img_np).float().unsqueeze(0) / 255.0
        lms_orig_tensor = torch.from_numpy(lms).float()

        lms_tensor = torch.from_numpy(lms).float()
        lms_for_gen = torch.flip(lms_tensor, dims=[-1])

        with torch.no_grad():
            hm_tensor = self.heatmap_generator(lms_for_gen.unsqueeze(0)).squeeze(0)
        hm_np = hm_tensor.numpy()

        input_img = img_np[np.newaxis, ...].astype(np.float32)
        input_hm = hm_np.astype(np.float32)

        if self.augment:
            input_img, input_hm = self.augmentor(input_img, input_hm)

        img_final = torch.from_numpy(input_img).float() / 255.0
        hm_final = torch.from_numpy(input_hm).float()

        return {"image": img_final, "label": hm_final, "image_orig": img_orig_tensor, "lm_orig": lms_orig_tensor}

def compute_mre(pred_heatmap, gt_heatmap, return_per_landmark=False):
    with torch.no_grad():
        if pred_heatmap.shape != gt_heatmap.shape:
            pred_heatmap = F.interpolate(pred_heatmap, size=gt_heatmap.shape[-2:], mode='bilinear', align_corners=False)

        B, K, H, W = pred_heatmap.shape
        pred_coords = []
        gt_coords = []
        CONFIDENCE_THRESHOLD = 0.1

        for b in range(B):
            for k in range(K):
                pred_hm = pred_heatmap[b, k].clone()
                pred_hm = torch.clamp(pred_hm, 0, 1)

                if pred_hm.max() > CONFIDENCE_THRESHOLD and pred_hm.sum() > 0:
                    pred_hm = pred_hm / pred_hm.max()
                    pred_hm_weighted = pred_hm ** 2
                    y_coords = torch.arange(H, dtype=torch.float32, device=pred_hm.device)
                    x_coords = torch.arange(W, dtype=torch.float32, device=pred_hm.device)
                    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
                    total_weight = pred_hm_weighted.sum()
                    pred_x = (pred_hm_weighted * xx).sum() / total_weight
                    pred_y = (pred_hm_weighted * yy).sum() / total_weight
                else:
                    flat_idx = pred_hm.flatten().argmax()
                    pred_y = (flat_idx // W).float()
                    pred_x = (flat_idx % W).float()

                gt_hm = gt_heatmap[b, k].clone()

                if gt_hm.max() > CONFIDENCE_THRESHOLD and gt_hm.sum() > 0:
                    gt_hm = gt_hm / gt_hm.max()
                    gt_hm_weighted = gt_hm ** 2
                    y_coords = torch.arange(H, dtype=torch.float32, device=gt_hm.device)
                    x_coords = torch.arange(W, dtype=torch.float32, device=gt_hm.device)
                    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
                    total_weight = gt_hm_weighted.sum()
                    gt_x = (gt_hm_weighted * xx).sum() / total_weight
                    gt_y = (gt_hm_weighted * yy).sum() / total_weight
                else:
                    flat_idx = gt_hm.flatten().argmax()
                    gt_y = (flat_idx // W).float()
                    gt_x = (flat_idx % W).float()

                pred_coords.append([pred_x, pred_y])
                gt_coords.append([gt_x, gt_y])

        pred_coords = torch.tensor(pred_coords, device=pred_heatmap.device)
        gt_coords = torch.tensor(gt_coords, device=gt_heatmap.device)
        distances = torch.sqrt(((pred_coords - gt_coords) ** 2).sum(dim=1))

        if return_per_landmark:
            distances = distances.view(B, K)
            return distances.mean().item(), distances.cpu().numpy()
        else:
            return distances.mean().item()

def save_debug_images(batch, save_path):
    try:
        img_orig = batch['image_orig'][0, 0].numpy()
        lm_orig = batch['lm_orig'][0].numpy()
        img_aug = batch['image'][0, 0].numpy()
        hm_aug = batch['label'][0].numpy()
        K, H, W = hm_aug.shape
        hm_flat = hm_aug.reshape(K, -1)
        idx = np.argmax(hm_flat, axis=1)
        lm_aug_y = idx // W
        lm_aug_x = idx % W
        plt.figure(figsize=(12, 12))
        plt.subplot(2, 2, 1)
        plt.imshow(img_orig, cmap='gray')
        plt.axis('off')
        plt.subplot(2, 2, 2)
        plt.imshow(img_orig, cmap='gray')
        plt.scatter(lm_orig[:, 0], lm_orig[:, 1], c='red', s=40, marker='x')
        plt.axis('off')
        plt.subplot(2, 2, 3)
        plt.imshow(img_aug, cmap='gray')
        plt.axis('off')
        plt.subplot(2, 2, 4)
        plt.imshow(img_aug, cmap='gray')
        plt.scatter(lm_aug_x, lm_aug_y, c='yellow', s=40, marker='+')
        plt.imshow(hm_aug.max(axis=0), cmap='jet', alpha=0.3)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except:
        pass

class AutoTrainer:
    def __init__(self, config: Dict[str, Any], exp_dir: str):
        self.config = config
        self.exp_dir = exp_dir
        train_params = config.get("training_params", {})
        self.device = torch.device(train_params.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.logger = logging.getLogger("StrategyExecutionModule")

        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None

        nb_landmarks = config.get("nb_landmarks", 2)
        img_size = tuple(config.get("data_config", {}).get("input_size", [512, 512]))
        sigma_value = config.get("sigma", 5)

        if isinstance(sigma_value, (list, tuple)):
            sigma_value = sigma_value[0] if sigma_value else 5

        if GaussianHeatmapGenerator is not None:
            self.heatmap_generator = GaussianHeatmapGenerator(
                nb_landmarks=nb_landmarks,
                sigmas=float(sigma_value),
                heatmap_size=img_size,
                learnable=False
            )
        self.target_size = img_size

    def _prepare_data(self):
        data_cfg = self.config.get("data_config", {})
        batch_size = data_cfg.get("batch_size", 8)
        train_files = self.config.get("train_files", [])
        val_files = self.config.get("val_files", [])
        train_ds = CustomLandmarkDataset(train_files, self.heatmap_generator, augment=True, target_size=self.target_size)
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
        if val_files:
            val_ds = CustomLandmarkDataset(val_files, self.heatmap_generator, augment=False, target_size=self.target_size)
            self.val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    def run_visual_check(self, num_samples=4):
        from matplotlib.gridspec import GridSpec
        vis_dir = os.path.join(self.exp_dir, "visual_checks")
        os.makedirs(vis_dir, exist_ok=True)
        train_files = self.config.get("train_files", [])
        if not train_files:
            return

        dataset_no_aug = CustomLandmarkDataset(train_files, self.heatmap_generator, augment=False, target_size=self.target_size)
        dataset_with_aug = CustomLandmarkDataset(train_files, self.heatmap_generator, augment=True, target_size=self.target_size)
        indices = torch.randperm(len(train_files))[:num_samples].tolist()

        for idx, sample_idx in enumerate(indices):
            try:
                batch_orig = dataset_no_aug[sample_idx]
                img_orig = batch_orig['image'].squeeze(0).numpy()
                lms_orig = batch_orig['lm_orig'].numpy()
                batch_aug = dataset_with_aug[sample_idx]
                img_aug = batch_aug['image'].squeeze(0).numpy()
                hm_aug = batch_aug['label'].numpy()
                nb_landmarks, H, W = hm_aug.shape
                
                lms_aug = []
                for k in range(nb_landmarks):
                    hm_k = hm_aug[k]
                    if hm_k.max() > 0.1:
                        hm_k_norm = hm_k / hm_k.max()
                        hm_k_weighted = hm_k_norm ** 2
                        yy, xx = np.mgrid[0:H, 0:W]
                        total_w = hm_k_weighted.sum()
                        x = (hm_k_weighted * xx).sum() / total_w
                        y = (hm_k_weighted * yy).sum() / total_w
                    else:
                        flat_idx = hm_k.argmax()
                        y = flat_idx // W
                        x = flat_idx % W
                    lms_aug.append([x, y])
                lms_aug = np.array(lms_aug)

                fig = plt.figure(figsize=(20, 5))
                gs = GridSpec(1, 4, figure=fig, wspace=0.3)

                ax1 = fig.add_subplot(gs[0, 0])
                ax1.imshow(img_orig, cmap='gray', vmin=0, vmax=1)
                ax1.scatter(lms_orig[:, 0], lms_orig[:, 1], c='red', s=80, marker='x', linewidths=3, alpha=0.9)
                ax1.axis('off')

                ax2 = fig.add_subplot(gs[0, 1])
                ax2.imshow(img_aug, cmap='gray', vmin=0, vmax=1)
                ax2.axis('off')

                ax3 = fig.add_subplot(gs[0, 2])
                ax3.imshow(img_aug, cmap='gray', vmin=0, vmax=1)
                ax3.scatter(lms_aug[:, 0], lms_aug[:, 1], c='lime', s=80, marker='+', linewidths=3, alpha=0.9)
                ax3.axis('off')

                ax4 = fig.add_subplot(gs[0, 3])
                ax4.imshow(img_aug, cmap='gray', vmin=0, vmax=1, alpha=0.6)
                hm_max = np.max(hm_aug, axis=0)
                im = ax4.imshow(hm_max, cmap='jet', alpha=0.6, vmin=0, vmax=1)
                ax4.axis('off')

                save_path = os.path.join(vis_dir, f'check_sample_{idx + 1}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close()
            except Exception:
                pass

        summary_path = os.path.join(vis_dir, 'CHECK_SUMMARY.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Sample count: {num_samples}\n")
            f.write(f"Landmark count: {self.config.get('nb_landmarks', 'N/A')}\n")
            f.write(f"Sigma value: {self.config.get('sigma', 'N/A')}\n")

    def _standardize_output(self, outputs, target_shape):
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        if isinstance(outputs, dict):
            if 'out' in outputs: outputs = outputs['out']
            elif 'output' in outputs: outputs = outputs['output']
            else: outputs = list(outputs.values())[0]

        if outputs.dim() == 3: outputs = outputs.unsqueeze(1)
        elif outputs.dim() == 2: outputs = outputs.unsqueeze(0).unsqueeze(0)

        if outputs.shape[-2:] != target_shape[-2:]:
            outputs = F.interpolate(outputs, size=target_shape[-2:], mode='bilinear', align_corners=False)

        if outputs.shape[1] != target_shape[1]:
            if not hasattr(self, 'channel_adapter'):
                self.channel_adapter = nn.Conv2d(outputs.shape[1], target_shape[1], kernel_size=1).to(outputs.device)
            outputs = self.channel_adapter(outputs)

        outputs = torch.sigmoid(outputs)
        return outputs

    def train(self):
        self.model = self.config["model_instance"].to(self.device)
        self.loss_function = self.config["training_params"]["loss_fn"]

        lr = self.config["training_params"]["lr"]
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        epochs = self.config["training_params"].get("epochs", 100)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)

        if not self.train_loader: self._prepare_data()

        scaler = torch.cuda.amp.GradScaler()
        best_mre = float('inf')
        best_epoch = -1
        HEATMAP_SCALE = 1.0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            epoch_mre = 0
            step = 0

            pbar = tqdm(self.train_loader, desc=f"Ep {epoch + 1}/{epochs}")
            for batch in pbar:
                step += 1
                inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                labels = labels * HEATMAP_SCALE

                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    raw_outputs = self.model(inputs)
                    outputs = self._standardize_output(raw_outputs, labels.shape)
                    loss = self.loss_function(outputs, labels)

                if torch.isnan(loss) or torch.isinf(loss):
                    self.optimizer.zero_grad()
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                scaler.step(self.optimizer)
                scaler.update()

                current_mre = compute_mre(outputs, labels)
                epoch_loss += loss.item()
                epoch_mre += current_mre
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "mre": f"{current_mre:.2f}"})

            if self.val_loader and (epoch + 1) % 5 == 0:
                val_loss, val_mre = self.validate(HEATMAP_SCALE)
                if (epoch + 1) % 10 == 0:
                    self._save_predictions(epoch + 1)

                self.scheduler.step(val_mre)
                if val_mre < best_mre:
                    best_mre = val_mre
                    best_epoch = epoch + 1
                    torch.save(self.model.state_dict(), os.path.join(self.exp_dir, "best_model.pth"))

            if (epoch + 1) % 50 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.exp_dir, f"ckpt_ep{epoch + 1}.pth"))

    def _save_predictions(self, epoch):
        self.model.eval()
        save_dir = os.path.join(self.exp_dir, "predictions")
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            batch = next(iter(self.val_loader))
            inputs = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            raw_outputs = self.model(inputs)
            outputs = self._standardize_output(raw_outputs, labels.shape)

            img = inputs[0, 0].cpu().numpy()
            gt_hm = labels[0].cpu().numpy()
            pred_hm = outputs[0].cpu().numpy()

            K = gt_hm.shape[0]
            gt_coords, pred_coords = [], []

            for k in range(K):
                gt_k = gt_hm[k]
                if gt_k.max() > 0.1:
                    gt_k_norm = gt_k / gt_k.max()
                    H, W = gt_k.shape
                    y_idx, x_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                    total = (gt_k_norm ** 2).sum()
                    gt_coords.append([((gt_k_norm ** 2) * x_idx).sum() / total, ((gt_k_norm ** 2) * y_idx).sum() / total])

                pred_k = pred_hm[k]
                if pred_k.max() > 0.1:
                    pred_k_norm = np.clip(pred_k, 0, 1) / np.clip(pred_k, 0, 1).max()
                    H, W = pred_k.shape
                    y_idx, x_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                    total = (pred_k_norm ** 2).sum()
                    pred_coords.append([((pred_k_norm ** 2) * x_idx).sum() / total, ((pred_k_norm ** 2) * y_idx).sum() / total])

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img, cmap='gray')
            axes[0].axis('off')

            axes[1].imshow(img, cmap='gray')
            axes[1].imshow(gt_hm.max(axis=0), cmap='jet', alpha=0.5)
            axes[1].axis('off')

            axes[2].imshow(img, cmap='gray')
            axes[2].imshow(pred_hm.max(axis=0), cmap='jet', alpha=0.5)
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"prediction_epoch_{epoch:03d}.png"), dpi=150)
            plt.close()

        self.model.train()

    def validate(self, scale_factor):
        self.model.eval()
        val_loss, val_mre, step = 0, 0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                step += 1
                inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device) * scale_factor

                with torch.cuda.amp.autocast():
                    raw_outputs = self.model(inputs)
                    outputs = self._standardize_output(raw_outputs, labels.shape)
                    loss = self.loss_function(outputs, labels)

                val_loss += loss.item()
                val_mre += compute_mre(outputs, labels)
        return val_loss / max(step, 1), val_mre / max(step, 1)
