# logger.py
import wandb
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Optional, List

class MetricLogger:
    """Base class for metric logging"""
    def __init__(self):
        self.metric_groups = {
            "train": ["loss", "accuracy", "f1"],
            "val": ["loss", "accuracy", "f1"],
            "global": ["epoch", "learning_rate"]
        }
        self.class_metrics = ["f1"]  # metrics to calculate per class

class ExperimentInfo:
    """Experiment information management class"""
    def __init__(self, cfg: DictConfig):
        self.data_config = {
            "augmentation": cfg.data.augmentation,
            "split_method": cfg.data.split_method,
            "test_size": cfg.data.get("test_size", 0.2),
        }
        
        self.model_config = {
            "name": cfg.model.name,
            "num_classes": cfg.model.num_classes,
            "pretrained": cfg.model.pretrained,
            "class_balancing": cfg.model.class_balancing.method,
            "domain_adaptation": cfg.model.domain_adaptation.method
        }
        
        self.train_config = {
            "epochs": cfg.train.epochs,
            "batch_size": cfg.train.batch_size,
            "img_size": cfg.train.img_size,
            "learning_rate": cfg.train.lr,
            "early_stopping": cfg.train.early_stopping.enabled,
            "mixed_precision": cfg.train.mixed_precision.enabled
        }
    
    def get_tags(self) -> List[str]:
        """Convert key information to tags"""
        return [
            f"model_{self.model_config['name']}",
            f"split_{self.data_config['split_method']}",
            f"aug_{self.data_config['augmentation']}",
            f"bal_{self.model_config['class_balancing']}"
        ]
    
    def get_run_name(self) -> str:
        """Generate experiment name"""
        return "_".join([
            f"Aug_{self.data_config['augmentation']}",
            f"Split_{self.data_config['split_method']}",
            f"Model_{self.model_config['name']}",
            f"CB_{self.model_config['class_balancing']}",
            f"DA_{self.model_config['domain_adaptation']}"
        ])
    
    def print_summary(self):
        """Print experiment configuration summary"""
        print("\n=== Experiment Configuration ===")
        print("\n[Data Configuration]")
        for k, v in self.data_config.items():
            print(f"{k:15s}: {v}")
            
        print("\n[Model Configuration]")
        for k, v in self.model_config.items():
            print(f"{k:15s}: {v}")
            
        print("\n[Training Configuration]")
        for k, v in self.train_config.items():
            print(f"{k:15s}: {v}")
        print("=" * 40 + "\n")

class WandbLogger(MetricLogger):
    def __init__(
        self,
        project_name: str,
        entity: str,
        experiment_info: ExperimentInfo,
        cfg: Optional[DictConfig] = None
    ):
        super().__init__()
        
        run_name = get_unique_run_name(
            project_name=project_name,
            base_name=experiment_info.get_run_name(),
            entity=entity
        )
        
        self.run = wandb.init(
            project=project_name,
            entity=entity,
            name=run_name,
            config=self._flatten_config(cfg) if cfg else None,
            tags=experiment_info.get_tags()
        )
        
        # Define metric groups
        for group in self.metric_groups:
            self.run.define_metric(f"{group}/*", step_metric="global/epoch")

    def _flatten_config(self, cfg: DictConfig) -> Dict[str, Any]:
        """Convert config to flattened dictionary"""
        if cfg is None:
            return {}
        
        # Keys to exclude
        exclude_keys = {
            'hydra', 'wandb', 'distribution', 
            'parameters', 'defaults', '_target_'
        }
        
        config_dict = OmegaConf.to_container(
            cfg,
            resolve=True,
            enum_to_str=True,
        )
        flattened = {}
        
        def _flatten(d: Dict, prefix: str = ''):
            if d is None:
                return
                
            for k, v in d.items():
                if k in exclude_keys:
                    continue
                    
                new_key = f"{prefix}.{k}" if prefix else k
                
                if isinstance(v, dict) and v:
                    if 'distribution' not in v:
                        _flatten(v, new_key)
                else:
                    if v is not None and not isinstance(v, dict):
                        flattened[new_key] = v
                        
        _flatten(config_dict)
        return flattened

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics"""
        if self.run is not None:
            # Filter out None values
            filtered_metrics = {
                k: v for k, v in metrics.items() 
                if v is not None
            }
            self.run.log(filtered_metrics, step=step)

    def finish(self):
        """Finish the wandb run"""
        if self.run is not None:
            self.run.finish()

def get_unique_run_name(project_name: str, base_name: str, entity: Optional[str] = None) -> str:
    """Generate unique run name"""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project_name}" if entity else project_name)
    
    existing_names = [run.name for run in runs if run.name]
    unique_name = base_name
    counter = 0
    
    while unique_name in existing_names:
        counter += 1
        unique_name = f"{base_name}_{counter}"
    
    return unique_name

def get_logger(cfg: DictConfig) -> Optional[WandbLogger]:
    """Create logger based on config"""
    if not cfg.logger.use_wandb:
        return None
    
    experiment_info = ExperimentInfo(cfg)
    experiment_info.print_summary()
    
    return WandbLogger(
        project_name=cfg.logger.project_name,
        entity=cfg.logger.entity,
        experiment_info=experiment_info,
        cfg=cfg
    )
