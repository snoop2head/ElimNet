"""
NAS search for lightweight model(but with empty weights).
- Author: Junghoon Kim, Jongsun Shin
"""
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataloader import create_dataloader
from src.model import Model
from src.utils.torch_utils import model_info, check_runtime
from src.trainer import TorchTrainer, count_model_params
from typing import Any, Dict, List, Tuple
from optuna.pruners import HyperbandPruner
from subprocess import _args_from_interpreter_flags
import argparse

# from optuna.integration.wandb import WeightsAndBiasesCallback
import pandas as pd
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
import yaml
import os

EPOCH = 100
DATA_PATH = "../data"  # type your data path here that contains test, train and val directories
RESULT_MODEL_PATH = "./result_model.pt"  # result model will be saved in this path


def search_hyperparam(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Search hyperparam from user-specified search space."""
    epochs = trial.suggest_int("epochs", low=20, high=20, step=2)
    img_size = trial.suggest_categorical("img_size", [96, 112, 168, 224])
    n_select = trial.suggest_int("n_select", low=0, high=6, step=2)
    batch_size = trial.suggest_int("batch_size", low=32, high=128, step=32)
    return {
        "EPOCHS": epochs,
        "IMG_SIZE": img_size,
        "n_select": n_select,
        "BATCH_SIZE": batch_size,
    }


def search_model(trial: optuna.trial.Trial) -> List[Any]:
    """Search model structure from user-specified search space."""
    model = []
    n_stride = 0
    MAX_NUM_STRIDE = 5
    UPPER_STRIDE = 2  # 5(224 example): 224, 112, 56, 28, 14, 7
    n_layers = trial.suggest_int("n_layers", 8, 12)
    stride = 1
    input_max = 64
    input_min = 32
    module_info = {}
    ### 몇개의 레이어를 쌓을지도 search하게 했습니다.
    for i in range(n_layers):
        out_channel = trial.suggest_int(f"{i+1}units", input_min, input_max)
        block = trial.suggest_categorical(
            f"m{i+1}",
            ["Conv", "DWConv", "InvertedResidualv2", "InvertedResidualv3", "MBConv"],
        )
        repeat = trial.suggest_int(f"m{i+1}/repeat", 1, 5)
        m_stride = trial.suggest_int(f"m{i+1}/stride", low=1, high=UPPER_STRIDE)
        if m_stride == 2:
            stride += 1
        if n_stride == 0:
            m_stride = 2

        if block == "Conv":
            activation = trial.suggest_categorical(f"m{i+1}/activation", ["ReLU", "Hardswish"])
            # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
            args = [out_channel, 3, m_stride, None, 1, activation]
        elif block == "DWConv":
            activation = trial.suggest_categorical(f"m{i+1}/activation", ["ReLU", "Hardswish"])
            # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
            args = [out_channel, 3, 1, None, activation]
        elif block == "InvertedResidualv2":
            c = trial.suggest_int(f"m{i+1}/v2_c", low=input_min, high=input_max, step=16)
            t = trial.suggest_int(f"m{i+1}/v2_t", low=1, high=4)
            args = [c, t, m_stride]
        elif block == "InvertedResidualv3":
            kernel = trial.suggest_int(f"m{i+1}/kernel_size", low=3, high=5, step=2)
            t = round(trial.suggest_float(f"m{i+1}/v3_t", low=1.0, high=6.0, step=0.1), 1)
            c = trial.suggest_int(f"m{i+1}/v3_c", low=input_min, high=input_max, step=8)
            se = trial.suggest_categorical(f"m{i+1}/v3_se", [0, 1])
            hs = trial.suggest_categorical(f"m{i+1}/v3_hs", [0, 1])
            # k t c SE HS s
            args = [kernel, t, c, se, hs, m_stride]
        elif block == "MBConv":
            kernel = trial.suggest_int(f"m{i+1}/kernel_size", low=3, high=5, step=2)
            c = trial.suggest_int(f"m{i+1}/efb0_c", low=input_min, high=input_max, step=8)
            # args=[_,c]

        in_features = out_channel
        model.append([repeat, block, args])
        if i % 2:
            input_max *= 2
            input_max = min(input_max, 160)
        module_info[f"block{i+1}"] = {"type": block, "repeat": repeat, "stride": stride}
    # last layer
    last_dim = trial.suggest_int("last_dim", low=128, high=1024, step=128)
    # We can setup fixed structure as well
    model.append([1, "Conv", [last_dim, 1, 1]])
    model.append([1, "GlobalAvgPool", []])
    model.append([1, "FixedConv", [6, 1, 1, None, 1, None]])
    return model, module_info


def objective(trial: optuna.trial.Trial, device, fp16) -> Tuple[float, int, float]:
    """Optuna objective.
    Args:
        trial
    Returns:
        float: score1(e.g. accuracy)
        int: score2(e.g. params)
    """
    model_config: Dict[str, Any] = {}
    model_config["input_channel"] = 3
    # img_size = trial.suggest_categorical("img_size", [32, 64, 128])
    img_size = 32
    model_config["INPUT_SIZE"] = [img_size, img_size]
    model_config["depth_multiple"] = trial.suggest_categorical(
        "depth_multiple", [0.25, 0.5, 0.75, 1.0]
    )
    model_config["width_multiple"] = trial.suggest_categorical(
        "width_multiple", [0.25, 0.5, 0.75, 1.0]
    )
    model_config["backbone"], module_info = search_model(trial)
    hyperparams = search_hyperparam(trial)

    model = Model(model_config, verbose=True)
    model.to(device)
    model.model.to(device)

    # check ./data_configs/data.yaml for config information
    data_config: Dict[str, Any] = {}
    data_config["DATA_PATH"] = DATA_PATH
    data_config["DATASET"] = "TACO"
    data_config["AUG_TRAIN"] = "randaugment_train"
    data_config["AUG_TEST"] = "simple_augment_test"
    data_config["AUG_TRAIN_PARAMS"] = {
        "n_select": hyperparams["n_select"],
    }
    data_config["AUG_TEST_PARAMS"] = None
    data_config["BATCH_SIZE"] = hyperparams["BATCH_SIZE"]
    data_config["VAL_RATIO"] = 0.8
    data_config["IMG_SIZE"] = hyperparams["IMG_SIZE"]
    data_config["INIT_LR"] = 0.0001
    data_config["EPOCHS"] = 100
    """이부분이 config를 저장하는 부분입니다. 위의 lr,fp16,epochs는 원래 함수에는 없지만
    config를 바로 사용할 수 있게 추가했습니다."""

    k = 1
    file_name = f"search_model/model_{k}.yaml"
    while os.path.exists(file_name):
        print(k)
        k += 1
        file_name = f"search_model/model_{k}.yaml"
    print(model_config)
    "model config와 data config를 저장"
    with open(f"search_model/model_{k}.yaml", "w") as outfile:
        yaml.dump(model_config, outfile)
    with open(f"search_model/data_{k}.yaml", "w") as outfile:
        yaml.dump(data_config, outfile)

    mean_time = check_runtime(
        model.model,
        [model_config["input_channel"]] + model_config["INPUT_SIZE"],
        device,
    )
    model_info(model, verbose=True)
    train_loader, val_loader, test_loader = create_dataloader(data_config)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        steps_per_epoch=len(train_loader),
        epochs=hyperparams["EPOCHS"],
        pct_start=0.05,
    )

    scaler = torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None

    trainer = TorchTrainer(
        model,
        criterion,
        optimizer,
        scheduler,
        device=device,
        verbose=1,
        model_path=RESULT_MODEL_PATH,
        scaler=scaler,
    )
    trainer.train(train_loader, hyperparams["EPOCHS"], val_dataloader=val_loader)
    loss, f1_score, acc_percent = trainer.test(model, test_dataloader=val_loader)
    params_nums = count_model_params(model)

    model_info(model, verbose=True)
    return f1_score, params_nums, mean_time


def get_best_trial_with_condition(optuna_study: optuna.study.Study) -> Dict[str, Any]:
    """Get best trial that satisfies the minimum condition(e.g. accuracy > 0.8).
    Args:
        study : Optuna study object to get trial.
    Returns:
        best_trial : Best trial that satisfies condition.
    """
    df = optuna_study.trials_dataframe().rename(
        columns={
            "values_0": "acc_percent",
            "values_1": "params_nums",
            "values_2": "mean_time",
        }
    )
    ## minimum condition : accuracy >= threshold
    threshold = 0.7
    minimum_cond = df.acc_percent >= threshold

    if minimum_cond.any():
        df_min_cond = df.loc[minimum_cond]
        ## get the best trial idx with lowest parameter numbers
        best_idx = df_min_cond.loc[
            df_min_cond.params_nums == df_min_cond.params_nums.min()
        ].acc_percent.idxmax()

        best_trial_ = optuna_study.trials[best_idx]
        print("Best trial which satisfies the condition")
        print(df.loc[best_idx])
    else:
        print("No trials satisfies minimum condition")
        best_trial_ = None

    return best_trial_


def tune(gpu_id, storage: str = None, fp16: bool = False):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif 0 <= gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
    sampler = optuna.samplers.MOTPESampler()
    if storage is not None:
        rdb_storage = optuna.storages.RDBStorage(url=storage)
    else:
        rdb_storage = None
    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize"],
        study_name="automl",
        sampler=sampler,
        storage=rdb_storage,
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, device, fp16), n_trials=10)
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trials:")
    best_trials = study.best_trials

    ## trials that satisfies Pareto Fronts
    for tr in best_trials:
        print(f"  value1:{tr.values[0]}, value2:{tr.values[1]}")
        for key, value in tr.params.items():
            print(f"    {key}:{value}")

    best_trial = get_best_trial_with_condition(study)
    print(best_trial)
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv("search_results.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna tuner.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("--storage", default="", type=str, help="Optuna database storage path.")
    parser.add_argument("--fp16", default=False, type=bool, help="train to fp16")
    args = parser.parse_args()
    tune(args.gpu, storage=args.storage if args.storage != "" else None, fp16=args.fp16)
