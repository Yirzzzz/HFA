from pathlib import Path
import random

ROOT = Path("./datasets")

SEEDS = [42, 512, 1024, 2048, 4096]
TOTAL = 30000

# Real 和 MM_CelebA 都不作为 Fake
EXCLUDE = {"real", "mm_celeba"}

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# 是否在每行路径后添加 Fake 标签 1
# 如果 Real.txt 只有路径，这里也保持 False
WITH_LABEL = False


def collect_images():
    """自动扫描所有伪造方法文件夹"""
    groups = {}

    for folder in sorted(ROOT.iterdir()):
        if not folder.is_dir():
            continue

        if folder.name.lower() in EXCLUDE:
            continue

        if folder.name.startswith("."):
            continue

        images = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )

        if images:
            groups[folder.name] = images
            print(f"{folder.name}: {len(images)} images")

    return groups


def balanced_sample(groups, seed):
    """按方法均衡分配数量，并进行随机抽样"""
    rng = random.Random(seed)

    methods = list(groups.keys())
    rng.shuffle(methods)

    counts = {m: 0 for m in methods}
    remaining = TOTAL

    if sum(len(v) for v in groups.values()) < TOTAL:
        raise ValueError("Fake 图片总量不足 30000 张")

    # 尽可能均匀分配，考虑部分方法数量不足
    while remaining > 0:
        available = [
            m for m in methods
            if counts[m] < len(groups[m])
        ]

        if not available:
            raise ValueError("可用图片不足")

        batch = remaining // len(available)

        if batch > 0:
            for m in available:
                n = min(batch, len(groups[m]) - counts[m])
                counts[m] += n
                remaining -= n
        else:
            for m in available[:remaining]:
                counts[m] += 1
            remaining = 0

    # 随机选取图片
    selected = []

    for method in methods:
        images = rng.sample(groups[method], counts[method])

        for image in images:
            relative_path = image.relative_to(ROOT).as_posix()

            if WITH_LABEL:
                selected.append(f"{relative_path} 1")
            else:
                selected.append(relative_path)

    # 打乱不同方法的图片顺序
    rng.shuffle(selected)

    return selected, counts


def main():
    groups = collect_images()

    if not groups:
        raise RuntimeError("没有找到 Fake 图片文件夹")

    print(f"\n共找到 {len(groups)} 种伪造方法\n")

    for seed in SEEDS:
        selected, counts = balanced_sample(groups, seed)

        output = ROOT / f"Fake_seed{seed}.txt"

        output.write_text(
            "\n".join(selected) + "\n",
            encoding="utf-8"
        )

        print(f"\nSeed {seed}")
        for method in sorted(counts):
            print(f"  {method}: {counts[method]}")

        print(f"Total: {len(selected)}")
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()