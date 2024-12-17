# logger.py
import wandb
from omegaconf import DictConfig, OmegaConf

class WandbLogger:
    def __init__(self, project_name, run_name):
        # run 객체를 변수로 들고 있게 함.
        self.run = wandb.init(project=project_name, name=run_name)
    
    def log(self, metrics, step=None):
        wandb.log(metrics, step=step)

    def finish(self):
        # 현재 wandb run을 명시적으로 종료
        wandb.finish()

def get_logger(cfg: DictConfig):
    logger = None
    use_wandb = cfg.logger.use_wandb
    project_name = cfg.logger.project_name
    run_name = cfg.logger.run_name

    if use_wandb:
        logger = WandbLogger(project_name=project_name, run_name=run_name)
        
        # 실험 추적을 위한 주요 정보 추가
        experiment_config = {
            # 모델 관련 정보
            "model.name": cfg.model.name,
            "model.pretrained": cfg.model.pretrained,
            "model.num_classes": cfg.model.num_classes,
            
            # 데이터 관련 정보
            "data.split_method": cfg.data.split_method,
            "data.n_splits": cfg.data.get("n_splits", 5),
            "data.fold_index": cfg.data.get("fold_index", 0),
            
            # 학습 관련 정보
            "train.batch_size": cfg.train.batch_size,
            "train.learning_rate": cfg.train.lr,
            "train.epochs": cfg.train.epochs,
            "train.image_size": cfg.train.img_size,
        }
        
        # 기존 config와 통합
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        flattened_dict = {}
        
        def flatten_dict(d, parent_key=''):
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    flatten_dict(v, new_key)
                else:
                    flattened_dict[new_key] = v
        
        flatten_dict(config_dict)
        
        # 실험 config와 기존 config 통합
        flattened_dict.update(experiment_config)
        wandb.config.update(flattened_dict)
    
    return logger
