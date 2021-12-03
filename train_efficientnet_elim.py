"""
Eliminating pretrained EfficientNetB0's top layers
- Author: shawnhyeonsoo, hihellohowareyou
- Reference: narumiruna's efficientnet-pytorch at https://github.com/narumiruna/efficientnet-pytorch
"""
import argparse
import os
from datetime import datetime
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from collections import OrderedDict

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.torch_utils import check_runtime, model_info
from src.modules.mbconv import MBConvGenerator
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from adamp import SGDP


model_urls = {
    "efficientnet_b0": "https://www.dropbox.com/s/9wigibun8n260qm/efficientnet-b0-4cfa50.pth?dl=1",
    "efficientnet_b1": "https://www.dropbox.com/s/6745ear79b1ltkh/efficientnet-b1-ef6aa7.pth?dl=1",
    "efficientnet_b2": "https://www.dropbox.com/s/0dhtv1t5wkjg0iy/efficientnet-b2-7c98aa.pth?dl=1",
    "efficientnet_b3": "https://www.dropbox.com/s/5uqok5gd33fom5p/efficientnet-b3-bdc7f4.pth?dl=1",
    "efficientnet_b4": "https://www.dropbox.com/s/y2nqt750lixs8kc/efficientnet-b4-3e4967.pth?dl=1",
    "efficientnet_b5": "https://www.dropbox.com/s/qxonlu3q02v9i47/efficientnet-b5-4c7978.pth?dl=1",
    "efficientnet_b6": None,
    "efficientnet_b7": None,
}


def train(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train."""
    # save model_config, data_config
    with open(os.path.join(log_dir, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, "model.yml"), "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

    model_instance = Model(model_config, verbose=True)
    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")
    state_dict = load_state_dict_from_url(model_urls["efficientnet_b0"])
    model_state_dict = model_instance.model.state_dict()
    model_state_keys = [i for i in model_instance.model.state_dict()]
    state_dict_keys = [i for i in state_dict]
    new_state_dict = OrderedDict()

    # Pretrained weights
    for i in range(len(model_state_keys) - 4):
        model_state = model_state_keys[i]
        pretrained_weight_state = state_dict_keys[i]
        new_state_dict[model_state] = state_dict[pretrained_weight_state]

    # Classifier
    for i in range(len(model_state_keys) - 2, len(model_state_keys)):
        model_state = model_state_keys[i]
        new_state_dict[model_state] = model_state_dict[model_state_keys[i]]

    model_instance.model.load_state_dict(new_state_dict, strict=False)

    if os.path.isfile(model_path):
        model_instance.model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
    model_instance.model.to(device)

    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Create optimizer, scheduler, criterion
    optimizer = SGDP(
        model_instance.model.parameters(), lr=data_config["INIT_LR"], momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=data_config["INIT_LR"],
        steps_per_epoch=len(train_dl),
        epochs=data_config["EPOCHS"],
        pct_start=0.05,
    )
    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"])
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
    )
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )

    # Create trainer
    trainer = TorchTrainer(
        model=model_instance.model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=model_path,
        verbose=1,
    )
    best_acc, best_f1 = trainer.train(
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=val_dl if val_dl else test_dl,
    )

    # evaluate model with test set
    model_instance.model.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc = trainer.test(
        model=model_instance.model, test_dataloader=val_dl if val_dl else test_dl
    )
    return test_loss, test_f1, test_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model",
        default="/opt/ml/data/model-optimization-level3-nlp-15/configs/model/effb0_.yaml",
        type=str,
        help="model config",
    )
    parser.add_argument(
        "--data", default="configs/data/taco.yaml", type=str, help="data config"
    )
    args = parser.parse_args()

    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)

    data_config["DATA_PATH"] = os.environ.get(
        "SM_CHANNEL_TRAIN", data_config["DATA_PATH"]
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp", "latest"))

    if os.path.exists(log_dir):
        modified = datetime.fromtimestamp(os.path.getmtime(log_dir + "/best.pt"))
        new_log_dir = (
            os.path.dirname(log_dir) + "/" + modified.strftime("%Y-%m-%d_%H-%M-%S")
        )
        os.rename(log_dir, new_log_dir)

    os.makedirs(log_dir, exist_ok=True)

    test_loss, test_f1, test_acc = train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=True,
        device=device,
    )
