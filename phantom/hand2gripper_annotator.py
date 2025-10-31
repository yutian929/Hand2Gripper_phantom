import logging
from enum import Enum
import os
from tqdm import tqdm
from joblib import Parallel, delayed  # type: ignore
import hydra
from omegaconf import DictConfig

from phantom.processors.base_processor import BaseProcessor
from phantom.processors.hand2gripper_annotator_processor import Hand2GripperAnnotator

logging.basicConfig(level=logging.WARNING, format="%(name)s - %(levelname)s - %(message)s")


def process_all_demos(cfg: DictConfig) -> None:
    
    base_processor = BaseProcessor(cfg)
    all_data_folders = base_processor.all_data_folders.copy()
    processor = Hand2GripperAnnotator(cfg)
    for data_sub_folder in tqdm(all_data_folders):
        try:
            processor.process_one_demo(data_sub_folder)
        except Exception as e:
            print(f"Error in Hand2Gripper annotating processing: {e}")
            if cfg.debug:
                raise


def main(cfg: DictConfig):
    process_all_demos(cfg)

@hydra.main(version_base=None, config_path="../configs", config_name="epic")
def hydra_main(cfg: DictConfig):
    """
    Main entry point using Hydra configuration.
    
    Example usage:
    - Process all demos with bbox: python process_data.py mode=bbox
    - Process single demo: python process_data.py mode=bbox demo_num=0
    - Use EPIC dataset: python process_data.py dataset=epic mode=bbox
    - Parallel processing: python process_data.py mode=bbox n_processes=4
    - Process multiple modes sequentially: python process_data.py mode=bbox,hand3d
    - Process with custom order: python process_data.py mode=hand3d,bbox,action
    - Process with bracket notation (use quotes): python process_data.py "mode=[bbox,hand3d]"
    """
    main(cfg)

if __name__ == "__main__":
    hydra_main()
