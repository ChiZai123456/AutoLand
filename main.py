import os
import sys
import yaml
import json
import argparse
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.figureprint import MedicalFingerprintExtractor
from modules.llm_client import AutoLandVLMAgent
from modules.config_parser import ConfigParser
from modules.trainer import AutoTrainer

def setup_logging(save_dir):
    log_file = os.path.join(save_dir, f"autoland_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("AutoLand-Pipeline")

def load_base_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if 'data_root' in cfg:
        data_root = cfg['data_root']
        for k, v in cfg.items():
            if isinstance(v, str) and "${data_root}" in v:
                cfg[k] = v.replace("${data_root}", data_root)
    return cfg

def make_file_list(img_dir, lbl_dir, lbl_suffix=".txt"):
    SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.nii.gz')
    all_files = os.listdir(img_dir)
    img_files = sorted([os.path.join(img_dir, f) for f in all_files if f.lower().endswith(SUPPORTED_IMAGE_FORMATS)])

    if not img_files:
        return []

    data_list = []
    missing_labels = []

    for img_p in img_files:
        base_name = os.path.splitext(os.path.basename(img_p))[0]
        if base_name.endswith('.nii'):
            base_name = os.path.splitext(base_name)[0]

        lbl_p = os.path.join(lbl_dir, base_name + lbl_suffix)
        if os.path.exists(lbl_p):
            data_list.append({"image": img_p, "label": lbl_p})
        else:
            missing_labels.append(base_name)

    return data_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--quick-test", action="store_true")
    args = parser.parse_args()

    try:
        base_cfg = load_base_config(args.config)
        if args.quick_test:
            base_cfg.setdefault('default_hyperparams', {})['epochs'] = 10
    except Exception as e:
        return

    task_name = base_cfg.get('task_name', 'experiment')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{task_name}_{timestamp}"
    exp_dir = os.path.join(base_cfg.get('output_dir', './output'), exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    logger = setup_logging(exp_dir)
    logger.info(f"Initializing AutoLand Workflow | Task: {task_name}")

    try:
        logger.info("\n=== [Stage 1/4] Physics Fingerprint Extraction ===")
        extractor = MedicalFingerprintExtractor(
            root_dir=base_cfg['train_img_dir'],
            output_dir=exp_dir
        )
        fingerprint_path = extractor.run()
        sampled_imgs_dir = os.path.join(exp_dir, "sampled_images")

        logger.info("\n=== [Stage 2/4] VLM Agent Strategy Generation ===")
        model_lib_path = base_cfg.get('model_library_path', 'model_structures.json')
        model_library = {}
        if os.path.exists(model_lib_path):
            with open(model_lib_path, 'r', encoding='utf-8') as f:
                model_library = json.load(f)
        
        agent = AutoLandVLMAgent(
            api_key=base_cfg.get('vlm_api_key', 'default_key'),
            model_name=base_cfg.get('vlm_model_name', 'vlm-large-instruct')
        )

        agent_strategy = agent.consult(
            task_desc=base_cfg,
            fingerprint_path=fingerprint_path,
            sampled_images_dir=sampled_imgs_dir,
            model_library=model_library
        )

        if not agent_strategy:
            raise RuntimeError("Agent failed to generate a valid strategy configuration.")

        nb_landmarks = base_cfg.get('nb_landmarks', 2)
        agent_strategy.setdefault('data_config', {})['nb_landmarks'] = nb_landmarks
        agent_strategy['nb_landmarks'] = nb_landmarks

        if 'model_config' in agent_strategy and 'params' in agent_strategy['model_config']:
            if agent_strategy['model_config']['params'].get('out_channels') != nb_landmarks:
                agent_strategy['model_config']['params']['out_channels'] = nb_landmarks

        agent_response_path = os.path.join(exp_dir, base_cfg.get('agent_response_filename', 'agent_strategy.json'))
        agent.save_response(agent_strategy, agent_response_path)
        logger.info(f"Agent Reasoning: {agent_strategy.get('reasoning', 'N/A')}")

        logger.info("\n=== [Stage 3/4] Strategy Parsing & Dynamic Instantiation ===")
        config_parser = ConfigParser(llm_config_dict=agent_strategy)
        device = base_cfg.get('default_hyperparams', {}).get('device', 'cuda')

        model_instance = config_parser.parse_model(device=device)
        training_params = config_parser.parse_training_params()
        data_config = config_parser.parse_data_config()

        if 'data_config' in base_cfg:
            base_data_config = base_cfg['data_config']
            if 'input_size' in base_data_config:
                data_config['input_size'] = base_data_config['input_size']
            if 'batch_size' in base_data_config:
                data_config['batch_size'] = base_data_config['batch_size']

        agent_sigma = config_parser.parse_sigma()
        logger.info(f"Instantiated Architecture: {model_instance.__class__.__name__}")

        logger.info("\n=== [Stage 4/4] Automated Execution (Strategy Execution Module) ===")
        train_files = make_file_list(base_cfg['train_img_dir'], base_cfg['train_lbl_dir'])
        val_files = make_file_list(base_cfg['test_img_dir'], base_cfg['test_lbl_dir'])

        if not train_files:
            raise RuntimeError("Data mapping failed: No valid training pairs.")

        final_sigma = agent_sigma if agent_sigma is not None else base_cfg.get('heatmap_sigma', 3)
        
        full_train_config = {
            "model_instance": model_instance,
            "training_params": {
                **base_cfg.get('default_hyperparams', {}),
                **training_params,
                "device": device
            },
            "data_config": data_config,
            "nb_landmarks": nb_landmarks,
            "sigma": final_sigma,
            "train_files": train_files,
            "val_files": val_files,
            "train_transforms": None,
            "val_transforms": None
        }

        trainer = AutoTrainer(full_train_config, exp_dir)
        try:
            trainer.run_visual_check()
        except:
            pass
            
        trainer.train()
        logger.info(f"\nWorkflow finalized. Best model artifact: {os.path.join(exp_dir, 'best_model.pth')}")

    except Exception as e:
        logger.error(f"Workflow interrupted: {e}")
        raise

if __name__ == "__main__":
    main()
