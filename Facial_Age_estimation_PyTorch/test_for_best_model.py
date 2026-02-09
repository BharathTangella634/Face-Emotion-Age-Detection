import torch
import torch.nn as nn
import torchmetrics as tm

from model2 import AgeEstimationModel
from functions import test
from custom_dataset_dataloader import valid_loader
from config import config


def calculate_accuracy_within_threshold(preds, targets, threshold=5):
    diff = torch.abs(preds - targets)
    correct = (diff <= threshold).float()
    return correct.mean().item() * 100


def main():

    device = config['device']

    # ---------------- Load Model ----------------
    model = AgeEstimationModel(
        input_dim=3,
        output_nodes=1,
        model_name='resnet',
        pretrain_weights='IMAGENET1K_V2'
    ).to(device)

    checkpoint_path = "/mnt/8b4bbd12-99b7-4ef1-9218-be56afd51a3d/Facial Emotion and Age Prediction/Facial_Age_estimation_PyTorch/checkpoints/epoch-29-loss_valid-4.61.pt"
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # ---------------- Loss & Metric ----------------
    loss_fn = nn.SmoothL1Loss()
    mae_metric = tm.MeanAbsoluteError().to(device)

    # ---------------- Evaluate ----------------
    loss_valid, mae_valid, preds, targets = test(
        model,
        valid_loader,
        loss_fn,
        mae_metric
    )

    accuracy_5 = calculate_accuracy_within_threshold(preds, targets, threshold=5)

    print("\n================ BEST MODEL RESULTS ================")
    print(f"Validation Loss        : {loss_valid:.4f}")
    print(f"Validation MAE         : {mae_valid:.4f}")
    print(f"Accuracy (±5 years)    : {accuracy_5:.2f}%")
    print("====================================================\n")


if __name__ == "__main__":
    main()
