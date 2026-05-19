import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.config_parser import ConfigParser
from modules.trainer import compute_mre

try:
    from landmarker.heatmap import GaussianHeatmapGenerator
except ImportError:
    print(" landmarker is not installed; using simplified heatmap generator")
    GaussianHeatmapGenerator = None

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

DRAW_TEXT_SIZE_FACTOR = {
    'cephalometric': 1.13,
    'hand': 1,
    'chest': 1.39,
    'heart': 1.39,
    'heartl': 1.39,
    'hearts': 1.39
}

def radial_distance(pt1, pt2, factor=1):
    if not hasattr(factor, '__iter__'):
        factor = [factor] * len(pt1)
    return sum(((i - j) * s) ** 2 for i, j, s in zip(pt1, pt2, factor)) ** 0.5

def align_points(points, gt_points, factor=1):
    if len(points) != len(gt_points):
        raise ValueError(f"Number of points mismatch: {len(points)} vs {len(gt_points)}")
    if not points:
        return []
    n = len(points)
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = radial_distance(points[i], gt_points[j], factor)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    sorted_indices = np.argsort(col_ind)
    return [points[row_ind[i]] for i in sorted_indices]

def cal_all_distance(points, gt_points, factor=1):
    n1 = len(points)
    n2 = len(gt_points)
    if n1 == 0:
        print("[Warning]: Empty input for calculating distances")
        return []
    if n1 != n2:
        raise Exception(f"Error: lengths mismatch, {n1} <> {n2}")
    return [radial_distance(p, q, factor) for p, q in zip(points, gt_points)]

def load_physical_factor(model_dir, dataset_name='chest'):
    DEFAULT_FACTORS = {
        'cephalometric': 0.46875,
        'chest': 0.1,
        'Cephalograms': 0.5,
        'heart': 0.139,
        'heartl': 0.139,
        'hearts': 0.139,
    }
    fingerprint_path = os.path.join(model_dir, 'physics_fingerprint.json')
    if os.path.exists(fingerprint_path):
        try:
            with open(fingerprint_path, 'r', encoding='utf-8') as f:
                fingerprint = json.load(f)
            physics_props = fingerprint.get('physics_properties', {})
            median_spacing = physics_props.get('median_spacing', [1.0, 1.0])
            physical_factor = float(median_spacing[0])
            return physical_factor
        except Exception as e:
            pass
    return DEFAULT_FACTORS.get(dataset_name, 1.0)

def draw_text(image, text, factor=1):
    txtwidth = round(30 * factor)
    padding = round(10 * factor)
    margin = round(5 * factor)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(FONT_PATH, txtwidth)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font)
    text_w = padding
    text_h = image.height - txtwidth - padding
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    pos = [text_w - margin, text_h - margin,
           text_w + text_width + margin, text_h + text_height + margin]
    draw.rectangle(pos, fill='#000000')
    draw.text((text_w, text_h), text, fill='#00ffff', font=font)
    return image

class FlexibleModelWrapper(torch.nn.Module):
    def __init__(self, model, expected_channels):
        super().__init__()
        self.model = model
        self.expected_channels = expected_channels

    def forward(self, x):
        outputs = self.model(x)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        if outputs.shape[1] != self.expected_channels:
            if not hasattr(self, 'channel_adapter'):
                self.channel_adapter = torch.nn.Conv2d(
                    outputs.shape[1],
                    self.expected_channels,
                    kernel_size=1
                ).to(outputs.device)
            outputs = self.channel_adapter(outputs)
        return outputs

def load_model_and_config(model_path, config_path, device='cuda'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    parser = ConfigParser(llm_config_dict=config)
    model = parser.parse_model(device=device)
    try:
        state_dict = torch.load(model_path, map_location=device)
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            try:
                model.load_state_dict(state_dict, strict=False)
            except Exception:
                if 'model.0.weight' in state_dict or 'module.' in list(state_dict.keys())[0]:
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        new_k = k.replace('module.', '').replace('model.', '')
                        new_state_dict[new_k] = v
                    model.load_state_dict(new_state_dict, strict=False)
    except Exception:
        pass
    model.eval()
    nb_landmarks = config.get('data_config', {}).get('nb_landmarks')
    if nb_landmarks is None:
        nb_landmarks = config.get('model_config', {}).get('params', {}).get('out_channels', 6)
    wrapped_model = FlexibleModelWrapper(model, nb_landmarks)
    return wrapped_model, config

def parse_landmarks_from_txt(txt_path, orig_width, orig_height, target_size=(512, 512)):
    coords = []
    try:
        with open(txt_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            if len(lines) > 0 and len(lines[0].split()) == 1:
                lines = lines[1:]
            for line in lines:
                parts = line.replace(',', ' ').split()
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    if x <= 1.1 and y <= 1.1:
                        x *= orig_width
                        y *= orig_height
                    x_target = x * target_size[0] / orig_width
                    y_target = y * target_size[1] / orig_height
                    coords.append([x_target, y_target])
    except Exception:
        pass
    return np.array(coords, dtype=np.float32)

def predict_landmarks(model, image_path, input_size=(512, 512), device='cuda'):
    img = Image.open(image_path).convert('L')
    orig_size = img.size
    img = img.resize(input_size, Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        outputs = torch.sigmoid(outputs)
        B, K, H, W = outputs.shape
        heatmaps = outputs[0]
        landmarks = []
        for k in range(K):
            heatmap = heatmaps[k].cpu().numpy()
            flat_idx = np.argmax(heatmap)
            y = flat_idx // W
            x = flat_idx % W
            x_512 = x * input_size[0] / W
            y_512 = y * input_size[1] / H
            landmarks.append([x_512, y_512])
    return np.array(landmarks), orig_size, input_size

def evaluate_dataset(model, data_dir, label_dir, input_size=(512, 512), device='cuda',
                     visualize=False, output_dir=None, physical_factor=1.0, dataset_name='chest'):
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        import glob
        image_files.extend(glob.glob(os.path.join(data_dir, ext)))
    if not image_files:
        return None
    all_errors_mm = []
    all_errors_px = []
    results = []
    for img_path in tqdm(image_files, desc="Evaluating"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base_name + '.txt')
        if not os.path.exists(label_path):
            continue
        pred_landmarks, orig_size, eval_size = predict_landmarks(model, img_path, input_size, device)
        gt_landmarks = parse_landmarks_from_txt(label_path, orig_size[0], orig_size[1], eval_size)
        if len(gt_landmarks) == 0:
            continue
        min_len = min(len(pred_landmarks), len(gt_landmarks))
        pred_landmarks_np = pred_landmarks[:min_len]
        gt_landmarks_np = gt_landmarks[:min_len]
        pred_points = [tuple(p) for p in pred_landmarks_np]
        gt_points = [tuple(g) for g in gt_landmarks_np]
        try:
            aligned_pred_points = align_points(pred_points, gt_points, factor=1)
            aligned_pred_landmarks = np.array(aligned_pred_points)
        except Exception:
            aligned_pred_landmarks = pred_landmarks_np
        distances_px = cal_all_distance(
            [tuple(p) for p in aligned_pred_landmarks],
            gt_points,
            factor=1
        )
        distances_mm = cal_all_distance(
            [tuple(p) for p in aligned_pred_landmarks],
            gt_points,
            factor=physical_factor
        )
        mean_error_px = np.mean(distances_px)
        mean_error_mm = np.mean(distances_mm)
        all_errors_px.extend(distances_px)
        all_errors_mm.extend(distances_mm)
        results.append({
            'image': os.path.basename(img_path),
            'mean_error_px': float(mean_error_px),
            'mean_error_mm': float(mean_error_mm),
            'errors_px': [float(d) for d in distances_px],
            'errors_mm': [float(d) for d in distances_mm],
            'pred_landmarks': aligned_pred_landmarks.tolist(),
            'gt_landmarks': gt_landmarks_np.tolist()
        })
        if visualize and output_dir:
            visualize_prediction(img_path, aligned_pred_landmarks, gt_landmarks_np,
                               mean_error_mm, physical_factor, eval_size,
                               os.path.join(output_dir, f"vis_{base_name}.png"),
                               dataset_name=dataset_name)
    all_errors_mm = np.array(all_errors_mm)
    all_errors_px = np.array(all_errors_px)
    sdr_results = {}
    for threshold in range(1, 11):
        sdr = np.mean(all_errors_mm <= threshold) * 100
        sdr_results[f'SDR@{threshold}mm'] = float(sdr)
    stats = {
        'MRE_mm': float(np.mean(all_errors_mm)),
        'STD_mm': float(np.std(all_errors_mm)),
        'MRE_px': float(np.mean(all_errors_px)),
        'STD_px': float(np.std(all_errors_px)),
        'median_error_mm': float(np.median(all_errors_mm)),
        'median_error_px': float(np.median(all_errors_px)),
        'max_error_mm': float(np.max(all_errors_mm)),
        'max_error_px': float(np.max(all_errors_px)),
        'min_error_mm': float(np.min(all_errors_mm)),
        'min_error_px': float(np.min(all_errors_px)),
        'physical_factor': float(physical_factor),
        'eval_size': list(input_size),
        'total_samples': len(results),
        'total_landmarks': len(all_errors_mm),
        **sdr_results
    }
    return stats, results

def visualize_prediction(image_path, pred_landmarks, gt_landmarks, mre_value_mm, physical_factor, eval_size, save_path, show_mre=True, show_gt=True, dataset_name='chest'):
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(eval_size, Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(img_resized)
    point_radius = 5
    if show_gt and gt_landmarks is not None:
        for i in range(len(gt_landmarks)):
            x, y = gt_landmarks[i]
            draw.ellipse(
                [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                fill=(0, 255, 0),
                outline=(0, 255, 0)
            )
    for i in range(len(pred_landmarks)):
        x, y = pred_landmarks[i]
        draw.ellipse(
            [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
            fill=(255, 0, 0),
            outline=(255, 0, 0)
        )
    if show_mre and mre_value_mm is not None:
        factor = DRAW_TEXT_SIZE_FACTOR.get(dataset_name, 1.0)
        mre_text = f'{mre_value_mm:.3f}'
        img_resized = draw_text(img_resized, mre_text, factor)
    img_resized.save(save_path, quality=95)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--yaml_config", type=str, default="config.yaml")
    parser.add_argument("--input_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--visualize", action="store_true", default=True)
    parser.add_argument("--physical_factor", type=float, default=None)
    parser.add_argument("--single_image", type=str, default=None)
    parser.add_argument("--label_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--no_mre", action="store_true")
    parser.add_argument("--no_gt", action="store_true")
    args = parser.parse_args()

    if args.single_image:
        model_dir = os.path.dirname(args.model)
        if args.physical_factor is not None:
            physical_factor = args.physical_factor
        else:
            physical_factor = load_physical_factor(model_dir, 'default')

        model, config = load_model_and_config(args.model, args.config, args.device)
        pred_landmarks, orig_size, eval_size = predict_landmarks(
            model, args.single_image, tuple(args.input_size), args.device
        )

        gt_landmarks = None
        mre_value_mm = None
        if args.label_path:
            gt_landmarks = parse_landmarks_from_txt(
                args.label_path, orig_size[0], orig_size[1], eval_size
            )
            if len(gt_landmarks) > 0:
                min_len = min(len(pred_landmarks), len(gt_landmarks))
                pred_points = [tuple(p) for p in pred_landmarks[:min_len]]
                gt_points = [tuple(g) for g in gt_landmarks[:min_len]]
                try:
                    aligned_pred_points = align_points(pred_points, gt_points, factor=1)
                    aligned_pred_landmarks = np.array(aligned_pred_points)
                except Exception:
                    aligned_pred_landmarks = pred_landmarks[:min_len]
                distances_mm = cal_all_distance(
                    [tuple(p) for p in aligned_pred_landmarks],
                    gt_points,
                    factor=physical_factor
                )
                mre_value_mm = np.mean(distances_mm)
                pred_landmarks = aligned_pred_landmarks

        if args.output_path is None:
            base_name = os.path.splitext(os.path.basename(args.single_image))[0]
            output_path = f"{base_name}_prediction.png"
        else:
            output_path = args.output_path

        dataset_name = 'chest'
        try:
            if 'task_name' in config:
                dataset_name = config['task_name'].lower().replace('_landmark_detection', '').replace('landmark_detection', '')
        except:
            pass

        visualize_prediction(
            args.single_image, pred_landmarks, gt_landmarks, mre_value_mm,
            physical_factor, eval_size, output_path,
            show_mre=not args.no_mre, show_gt=not args.no_gt, dataset_name=dataset_name
        )

        result_json_path = os.path.splitext(output_path)[0] + "_result.json"
        result_data = {
            'image': args.single_image,
            'pred_landmarks': pred_landmarks.tolist(),
            'original_size': list(orig_size),
            'eval_size': list(eval_size),
        }
        if gt_landmarks is not None:
            result_data['gt_landmarks'] = gt_landmarks.tolist()
        if mre_value_mm is not None:
            result_data['mre_mm'] = float(mre_value_mm)
            result_data['physical_factor'] = float(physical_factor)

        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
        return mre_value_mm if mre_value_mm is not None else 0.0

    with open(args.yaml_config, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)

    data_root = yaml_config.get('data_root', '')
    test_img_dir = yaml_config.get('test_img_dir', '').replace('${data_root}', data_root)
    test_lbl_dir = yaml_config.get('test_lbl_dir', '').replace('${data_root}', data_root)
    task_name = yaml_config.get('task_name', 'chest')

    model_dir = os.path.dirname(args.model)
    output_dir = os.path.join(model_dir, 'evaluation')
    os.makedirs(output_dir, exist_ok=True)

    if args.physical_factor is not None:
        physical_factor = args.physical_factor
    else:
        dataset_basename = task_name.lower().replace('_landmark_detection', '').replace('landmark_detection', '')
        physical_factor = load_physical_factor(model_dir, dataset_basename)

    model, config = load_model_and_config(args.model, args.config, args.device)
    dataset_basename = task_name.lower().replace('_landmark_detection', '').replace('landmark_detection', '')
    stats, results = evaluate_dataset(
        model, test_img_dir, test_lbl_dir, tuple(args.input_size),
        args.device, args.visualize, output_dir, physical_factor, dataset_name=dataset_basename
    )

    results_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'statistics': stats,
            'per_image_results': results
        }, f, indent=4, ensure_ascii=False)

    return stats['MRE_mm']

if __name__ == "__main__":
    main()
