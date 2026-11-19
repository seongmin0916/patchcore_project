import random
from pathlib import Path
from project1.imageloading import collect_image_list
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt

def split_records(records,valid_ratio=0.2,seed=42):
    train = [r.copy() for r in records if r["split"]=="train"]
    test = [r.copy() for r in records if r["split"]=="test"]


    if not 0 < valid_ratio < 1:
        raise ValueError("검증 비율은 0과 1사이여야 합니다.")
    if len(train) < 2:
        raise ValueError("학습 이미지 부족")
    if any(r["label"] != 0 for r in train):
        raise ValueError("학습 데이터에는 정상 이미지만 가능")

    train.sort(key=lambda r: r["image_path"])
    random.Random(seed).shuffle(train)

    count = max(1,int(len(train) * valid_ratio))
    valid = train[:count]
    train = train[count:]

    for r in valid :
        r["split"] = "valid"

    print(f"학습: {len(train)}장, 검증 : {len(valid)}장, 테스트 : {len(test)}")
    return train, valid, test

def read_image(record,debug=False) :
    image_path = record["image_path"]

    try:
        with Image.open(image_path) as source :
            image = source.convert("RGB")
    except (OSError, ValueError) as error :
        raise ValueError(f"이미지 읽기 오류: {image_path}") from error

    if debug :
        print(f"경로: {image_path}")
        print(f"크기: {image.size}, 채널:{image.mode}")

    return image


def preprocess_image(record, size=224) :
    transform = transforms.Compose([transforms.Resize((size,size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean = [0.485,0.456,0.406],
                                        std = [0.229,0.224,0.225]
                                    )])
    return transform(read_image(record))


class ProductDataSet(Dataset) :
    def __init__(self,records,size=224) :
        self.records = records
        self.size = size

    def __len__(self) :
        return len(self.records)

    def __getitem__(self,index) :
        record = self.records[index]

        return {"image" : preprocess_image(record,self.size),
                "label": record["label"],
                "image_path": record["image_path"]}

def make_loaders(train,valid,test,size=224,batch_size=32,seed=42):
    return tuple(
        DataLoader(ProductDataSet(records,size),
                   batch_size=batch_size,
                   shuffle=(index==0),
                   generator=torch.Generator().manual_seed(seed + index),
                   num_workers=0)
        for index, records in enumerate((train,valid,test)))



def check_loader(loader) : 
    samples = {}
    total = 0

    for batch in loader:
        images = batch["image"]
        labels = batch["label"]

        assert images.ndim == 4 and images.shape[1] == 3
        assert labels.shape == (len(images),)
        assert torch.isfinite(images).all()
        assert ((labels == 0) | (labels == 1)).all()

        for label in (0,1) :
            indices = (labels == label).nonzero(as_tuple =True)[0]
            if label not in samples and len(indices) :
                i = indices[0].item()
                samples[label] = images[i].detach().cpu().clone()
        total += len(images)

    if total == 0:
        raise ValueError("검사할 이미지x")

    print(f"검사 완료: {total}장")

    for label, image in sorted(samples.items()):
        mean = image.new_tensor([0.485,0.456,0.406])[:,None,None]
        std = image.new_tensor([0.229,0.224,0.225]) [:,None,None]
        image = (image * std + mean).clamp(0,1).permute(1,2,0)

        fig, ax = plt.subplots(figsize=(3,3))
        ax.imshow(image)
        ax.set_title(f"image: label={label}")
        ax.axis("off")

        plt.tight_layout()
        plt.show()
        plt.close(fig)


if __name__ == "__main__" :
    data_dir = Path(__file__).resolve().parent.parent / "mvtec_anomaly_detection"
    datasets = {}

    for product_dir in sorted(data_dir.iterdir()) :
        if not product_dir.is_dir() :
            continue

        print(f"\n상품: {product_dir.name}")
        records = collect_image_list(product_dir)
        datasets[product_dir.name] = split_records(records)
