from pathlib import Path


def collect_image_list(product_dir: str | Path) -> list[dict] :

    product_dir = Path(product_dir).expanduser().resolve()
    if not product_dir.is_dir():
        raise FileNotFoundError(f"상품 폴더를 찾을 수 없습니다 : {product_dir}")

    for split in ("train","test") :
        if not(product_dir/split).is_dir():
            raise ValueError(f"{product_dir} 안에 {split} 폴더가 없습니다.")


    image_extensions = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    records = []

    for split in ("train","test"):
        split_dir = product_dir/split
        for type_dir in sorted(split_dir.iterdir()):
            if not type_dir.is_dir():
                continue

            defect_type = type_dir.name
            label = 0 if defect_type == "good" else 1

            for image_path in sorted(type_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in image_extensions:
                    records.append({
                        "image_path": str(image_path),
                        "product": product_dir.name,
                        "split": split,
                        "label": label,
                        "defect_type": defect_type
                    })
    if not records:
        raise ValueError(f"이미지 파일을 찾을 수 없습니다: {product_dir}")

    return records

