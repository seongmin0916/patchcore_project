import os
import random

import numpy as np
import torch
from torch import nn
from anomalib.models.image.patchcore.torch_model import PatchcoreModel


def set_seed(seed=42):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_model(pretrained=True, seed=42):
    set_seed(seed)

    model = PatchcoreModel(
        backbone="wide_resnet50_2.racm_in1k",
        layers=["layer2", "layer3"],
        pre_trained=pretrained,
        num_neighbors=9
    )

    model.requires_grad_(False)
    model.eval()
    return model


@torch.no_grad()
def fit_memory(model, train_loader, ratio=0.1, device="cpu"):
    if not 0 < ratio <= 1:
        raise ValueError("ratio는 0보다 크고 1 이하여야 합니다.")

    model.to(device).eval()
    embeddings = []

    for index, batch in enumerate(train_loader, start=1):
        if (batch["label"] != 0).any():
            raise ValueError("정상 이미지만 메모리에 저장할 수 있습니다.")

        images = batch["image"].to(device)
        features = model.feature_extractor(images)
        features = {
            name: model.feature_pooler(value)
            for name, value in features.items()
        }

        embedding = model.generate_embedding(features)
        embedding = model.reshape_embedding(embedding)

        if not torch.isfinite(embedding).all():
            raise ValueError("특징에 NaN 또는 Inf가 있습니다.")

        embeddings.append(embedding.cpu())
        print(f"Features: batch {index}/{len(train_loader)}", flush=True)

    if not embeddings:
        raise ValueError("학습 데이터가 없습니다.")

    model.cpu()
    patch_count = sum(len(embedding) for embedding in embeddings)
    model.embedding_store.clear()
    model.embedding_store.extend(embeddings)
    del embeddings
    print(f"Coreset selection: {patch_count} patches, ratio={ratio}", flush=True)

    if int(patch_count * ratio) < model.num_neighbors:
        raise ValueError("저장할 특징 수가 이웃 수보다 적습니다.")

    model.subsample_embedding(sampling_ratio=ratio)
    model.to(device).eval()

    print("정상 특징 저장소:", tuple(model.memory_bank.shape))
    return model


@torch.no_grad()
def predict_scores(model, loader):
    device = next(model.parameters()).device
    model.eval()
    scores, labels = [], []

    for index, batch in enumerate(loader, start=1):
        output = model(batch["image"].to(device))
        score = output.pred_score.reshape(-1)

        if not torch.isfinite(score).all():
            raise ValueError("예측 점수에 NaN 또는 Inf가 있습니다.")

        scores.append(score.cpu())
        labels.append(batch["label"].cpu())
        print(f"Scores: batch {index}/{len(loader)}", flush=True)

    if not scores:
        raise ValueError("예측 데이터가 없습니다.")

    return torch.cat(scores), torch.cat(labels)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
        )

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or any(size % 8 for size in images.shape[-2:]):
            raise ValueError("이미지는 RGB 배치이며 높이와 너비가 8의 배수여야 합니다.")
        return self.decoder(self.encoder(images))


def epoch_loss(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_images = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            if (batch["label"] != 0).any():
                raise ValueError("학습·검증에는 정상 이미지만 사용합니다.")
            images = batch["image"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            restored = model(images)
            loss = nn.functional.mse_loss(restored, images)
            if not torch.isfinite(loss):
                raise ValueError("손실에 NaN 또는 Inf가 발생했습니다.")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(images)
            total_images += len(images)

    if total_images == 0:
        raise ValueError("학습 또는 검증 데이터가 비어 있습니다.")
    return total_loss / total_images


def train_autoencoder(model, train_loader, valid_loader, device,
                      max_epochs=20, patience=3, learning_rate=0.001, min_delta=0.00001):
    if max_epochs < 1 or patience < 1 or learning_rate <= 0 or min_delta < 0:
        raise ValueError("epoch·patience·학습률은 양수, min_delta는 0 이상이어야 합니다.")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_weights = None
    waiting = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = epoch_loss(model, train_loader, device, optimizer)
        valid_loss = epoch_loss(model, valid_loader, device)
        print(f"Epoch {epoch:02d}/{max_epochs} | train MSE {train_loss:.6f} | valid MSE {valid_loss:.6f}", flush=True)
        if valid_loss < best_loss - min_delta:
            best_loss = valid_loss
            best_weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            waiting = 0
        else:
            waiting += 1
            if waiting >= patience:
                print("Early stopping", flush=True)
                break

    model.load_state_dict(best_weights)
    return model.eval()


@torch.no_grad()
def predict_autoencoder(model, loader):
    device = next(model.parameters()).device
    model.eval()
    scores, labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        restored = model(images)
        score = (restored - images).square().mean(dim=(1, 2, 3))
        if not torch.isfinite(score).all():
            raise ValueError("예측 점수에 NaN 또는 Inf가 발생했습니다.")
        scores.append(score.cpu())
        labels.append(batch["label"].cpu())
    if not scores:
        raise ValueError("예측 데이터가 비어 있습니다.")
    return torch.cat(scores), torch.cat(labels)


