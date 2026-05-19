import os
import json
import inspect
import numpy as np
import torch
import torch.nn as nn
from glob import glob
from tqdm import tqdm
from monai.transforms import LoadImage
import monai.networks.nets as monai_nets

try:
    import medical_dl_library.models as custom_nets
    HAS_CUSTOM_LIB = True
except ImportError:
    HAS_CUSTOM_LIB = False

class MedicalImageFingerprint:
    def __init__(self, dataset_dir: str, suffix: str = ".png"):
        self.dataset_dir = dataset_dir
        self.suffix = suffix
        self.files = sorted(glob(os.path.join(dataset_dir, f"*{suffix}")))
        self.loader = LoadImage(image_only=False, ensure_channel_first=True)

    def _to_serializable(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def analyze_single_case(self, file_path):
        try:
            data, meta = self.loader(file_path)
            spatial_shape = list(data.shape[1:])
            n_dims = len(spatial_shape)
            spacing = meta.get('pixdim', [1.0] * (n_dims + 1))[1:1 + n_dims].tolist() if isinstance(meta, dict) else [1.0] * n_dims
            data_np = data.numpy()
            
            return {
                "n_dims": n_dims,
                "shape": spatial_shape,
                "spacing": spacing,
                "intensity": {
                    "min": self._to_serializable(data_np.min()),
                    "max": self._to_serializable(data_np.max()),
                    "mean": self._to_serializable(data_np.mean()),
                }
            }
        except:
            return None

    def run(self):
        all_shapes, all_spacings, all_intensities = [], [], []
        scan_limit = min(len(self.files), 50)
        
        for i in tqdm(range(scan_limit)):
            info = self.analyze_single_case(self.files[i])
            if info:
                all_shapes.append(info['shape'])
                all_spacings.append(info['spacing'])
                all_intensities.append(info['intensity'])

        if not all_shapes: 
            return {}

        all_shapes = np.array(all_shapes)
        all_spacings = np.array(all_spacings)
        
        return {
            "dataset_summary": {"total_samples": len(self.files), "file_type": self.suffix},
            "physics_properties": {
                "spatial_dims": int(all_shapes.shape[1]),
                "median_shape": np.median(all_shapes, axis=0).astype(int).tolist(),
                "median_spacing": np.median(all_spacings, axis=0).tolist(),
            },
            "intensity_statistics": {
                "global_mean": self._to_serializable(np.mean([i['mean'] for i in all_intensities])),
                "global_max": self._to_serializable(max([i['max'] for i in all_intensities]))
            }
        }

class ModelLibraryInspector:
    def __init__(self):
        self.available_models = {}

    def _get_class_signature(self, cls_obj):
        try:
            sig = inspect.signature(cls_obj.__init__)
            return {
                name: {
                    "default": str(param.default) if param.default != inspect.Parameter.empty else "REQUIRED",
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any"
                }
                for name, param in sig.parameters.items() if name != 'self'
            }
        except:
            return {}

    def _scan_module_architectures(self, module_obj, module_name):
        module_dict = {}
        for name, obj in inspect.getmembers(module_obj):
            if inspect.isclass(obj) and issubclass(obj, nn.Module) and not name.startswith("_"):
                module_dict[name] = self._get_class_signature(obj)
        self.available_models[module_name] = module_dict

    def generate_report(self):
        self._scan_module_architectures(monai_nets, "MONAI_Core")
        if HAS_CUSTOM_LIB:
            self._scan_module_architectures(custom_nets, "AutoLand_Medical_Library")
        return self.available_models

class MedicalFingerprintExtractor:
    def __init__(self, root_dir, output_dir):
        self.root_dir = root_dir
        self.output_dir = output_dir

    def run(self):
        data_extractor = MedicalImageFingerprint(self.root_dir, suffix=".png")
        data_fingerprint = data_extractor.run()

        model_inspector = ModelLibraryInspector()
        model_fingerprint = model_inspector.generate_report()

        full_report = {
            "data_fingerprint": data_fingerprint,
            "medical_dl_library": model_fingerprint
        }

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, "physics_fingerprint.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=4)

        return output_path
