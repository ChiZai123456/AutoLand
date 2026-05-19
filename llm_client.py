import os
import json
import base64
import logging
import glob
from openai import OpenAI
from typing import Dict, Any, Optional

try:
    from modules.config_parser import construct_english_prompt
except ImportError:
    from config_parser import construct_english_prompt

class AutoLandVLMAgent:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", model_name: str = "vlm-large-instruct"):
        if not api_key:
            raise ValueError("API Key is required.")
            
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def consult(self,
                task_desc: Dict[str, Any],
                fingerprint_path: str,
                sampled_images_dir: str,
                model_library: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

        if not os.path.exists(fingerprint_path):
            return None

        with open(fingerprint_path, 'r', encoding='utf-8') as f:
            fingerprint = json.load(f)

        messages = [{"role": "system", "content": "You are the AutoLand Medical Analysis Agent. You strictly output JSON."}]
        user_content = []

        base_prompt = construct_english_prompt(task_desc, fingerprint, [])
        model_context_str = ""

        if model_library:
            simplified_lib = {}
            KEYWORDS = [
                "SpatialConfiguration",
                "HighResNet",
                "Attention",
                "UNet",
                "UNETR",
                "SegResNet"
            ]

            is_flat_structure = False
            first_val = next(iter(model_library.values())) if model_library else None
            if isinstance(first_val, dict) and "parameters" in first_val:
                is_flat_structure = True

            if is_flat_structure:
                for model_name, details in model_library.items():
                    if isinstance(details, dict) and any(k in model_name for k in KEYWORDS):
                        simplified_lib[model_name] = details.get("parameters", {})
            else:
                for lib_name, models in model_library.items():
                    if isinstance(models, dict):
                        simplified_lib[lib_name] = {}
                        for model_name, details in models.items():
                            if isinstance(details, dict) and any(k in model_name for k in KEYWORDS):
                                simplified_lib[lib_name][model_name] = details.get("parameters", {})

            model_context_str = f"""
            \n\n### 5. AVAILABLE MEDICAL DL LIBRARY
            You MUST choose a model architecture from the list below.
            
            {json.dumps(simplified_lib, indent=2)}
            
            **INSTRUCTION**: 
            1. **Selection**: Select the most suitable model from the list above.
            2. **Parameters**: Fill in the 'params' in your JSON output using the EXACT parameter names listed above. 
            3. **Do NOT** invent parameters.
            """
            self.logger.info(" Loaded specific architectural constraints from Medical DL Library.")
        else:
            self.logger.warning(" Library not provided; proceeding with unconstrained generation.")

        full_text_prompt = base_prompt + model_context_str
        user_content.append({"type": "text", "text": full_text_prompt})

        if os.path.exists(sampled_images_dir):
            img_files = sorted(glob.glob(os.path.join(sampled_images_dir, "*.png")))[:3]
            if img_files:
                for img_p in img_files:
                    base64_image = self._encode_image(img_p)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    })

        messages.append({"role": "user", "content": user_content})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=4096,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            parsed_config = json.loads(content)
            parsed_config = self._validate_and_fix_config(parsed_config, task_desc, fingerprint)
            return parsed_config

        except Exception as e:
            return self._get_fallback_config(task_desc, fingerprint)

    def _validate_and_fix_config(self, config, task_desc, fingerprint):
        if 'reasoning' not in config or config['reasoning'] is None:
            config['reasoning'] = "Model recommendation generated by AutoLand Agent."

        if 'model_config' not in config or config['model_config'] is None:
            config['model_config'] = {
                "name": "UNet",
                "params": {
                    "spatial_dims": fingerprint.get("physics_properties", {}).get("spatial_dims", 2),
                    "in_channels": 1,
                    "out_channels": task_desc.get('nb_landmarks', 2),
                    "channels": (32, 64, 128, 256),
                    "strides": (2, 2, 2)
                }
            }

        if 'data_config' not in config or config['data_config'] is None:
            median_shape = fingerprint.get("physics_properties", {}).get("median_shape", [512, 512])
            config['data_config'] = {
                "input_size": median_shape[-2:] if len(median_shape) >= 2 else [512, 512],
                "batch_size": 8,
                "heatmap_sigma": 30
            }

        if 'training_params' not in config or config['training_params'] is None:
            config['training_params'] = {
                "loss": "ModifiedFocalLoss",
                "optimizer": "AdamW",
                "lr": 1e-3
            }

        if 'model_config' in config and 'params' in config['model_config']:
            expected_channels = task_desc.get('nb_landmarks', 2)
            if 'out_channels' in config['model_config']['params']:
                if config['model_config']['params']['out_channels'] != expected_channels:
                    config['model_config']['params']['out_channels'] = expected_channels

        return config

    def _get_fallback_config(self, task_desc, fingerprint):
        physics = fingerprint.get("physics_properties", {})
        spatial_dims = physics.get("spatial_dims", 2)
        median_shape = physics.get("median_shape", [512, 512])

        return {
            "reasoning": "Agentic fallback triggered; employing baseline parameterization.",
            "model_config": {
                "name": "UNet",
                "params": {
                    "spatial_dims": spatial_dims,
                    "in_channels": 1,
                    "out_channels": task_desc.get('nb_landmarks', 2),
                    "channels": (64, 128, 256, 512),
                    "strides": (2, 2, 2)
                }
            },
            "data_config": {
                "input_size": median_shape[-2:] if len(median_shape) >= 2 else [512, 512],
                "batch_size": 8,
                "heatmap_sigma": 30
            },
            "training_params": {
                "loss": "MSE",
                "optimizer": "AdamW",
                "lr": 1e-3
            }
        }

    def save_response(self, config: Dict, save_path: str):
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception:
            pass
