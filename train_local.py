# Local Training Script
import os
import sys
import re
import torch
from multiprocessing import freeze_support

# Paths
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = SCRIPT_DIR
DATASET_ROOT = os.path.join(REPO_ROOT, "datasets", "skyfusion.v1i.yolov11")
WEIGHTS_DIR  = os.path.join(REPO_ROOT, "datasets", "weights") # put best.pt / last.pt here
LAST_PT      = os.path.join(WEIGHTS_DIR, "last.pt")
BEST_PT      = os.path.join(WEIGHTS_DIR, 'best.pt')
DATASET_YAML = os.path.join(REPO_ROOT, "ultralytics", "cfg", "datasets", "skyfusion.yaml")

if __name__ == "__main__":
    freeze_support()

    print("=== Environment Check ===")
    print(f"Python  : {sys.version}")
    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM    : {vram:.1f} GB")

    # Sanity Checks
    print("\n=== Path Check ===")
    for label, path in [("Repo root",    REPO_ROOT),
                        ("last.pt",      LAST_PT),
                        ("dataset yaml", DATASET_YAML)]:
        status = "OK" if os.path.exists(path) else "NOT FOUND"
        print(f"  [{status}] {label}: {path}")

    if not os.path.exists(LAST_PT):
        raise FileNotFoundError(
            f"\nlast.pt not found at: {LAST_PT}\n"
            f"Make sure you placed last.pt inside: {WEIGHTS_DIR}"
        )

    if not os.path.exists(DATASET_YAML):
        raise FileNotFoundError(
            f"\nskyfusion.yaml not found at: {DATASET_YAML}\n"
            "Make sure the repo is cloned correctly."
        )

    # Patch skyfusion.yaml
    print("\n=== Patching skyfusion.yaml ===")
    with open(DATASET_YAML, "r") as f:
        yaml_content = f.read()
    local_dataset_path = DATASET_ROOT.replace("\\", "/")
    yaml_content_patched = re.sub(
        r"^path\s*:.*$",
        f"path: {local_dataset_path}",
        yaml_content,
        flags=re.MULTILINE
    )

    for split in ["train", "valid", "test"]:
        yaml_content_patched = re.sub(
            rf"^{split}\s*:.*$",
            f"{split}: {split}/images",
            yaml_content_patched,
            flags=re.MULTILINE
        )

    with open(DATASET_YAML, "w") as f:
        f.write(yaml_content_patched)

    print(f"  Set dataset path to: {local_dataset_path}")
    print("  train/valid/test → <split>/images")
    print("  (skyfusion.yaml patched successfully)")

    # Currently set to RESUME training.
    print("\n=== Starting Training ===")

    os.chdir(REPO_ROOT)
    sys.path.insert(0, REPO_ROOT)

    from ultralytics import YOLO

    model = YOLO(LAST_PT)   # resume from last.pt

    results = model.train(
        data=DATASET_YAML,
        epochs=60,
        imgsz=640,
        batch=4,           
        device=0,         
        project=os.path.join(REPO_ROOT, "runs", "final_project"),
        name="spdema_yolo26s_local",
        resume=True,       
        save=True,       
        save_period=1,    
        flipud=0.5,
        mosaic=1.0,
        close_mosaic=15,
        mixup=0.0,
        copy_paste=0.3,
        optimizer="auto",
        cos_lr=True,
        patience=30,
        plots=True,
    )

    print("\n=== Training Complete ===")
    print(f"Results saved to: {results.save_dir}")

    best_pt = os.path.join(BEST_PT)
    print(f"\n=== Validating best.pt ===")
    print(f"Loading: {best_pt}")

    best_model = YOLO(best_pt)
    metrics = best_model.val(
        data=DATASET_YAML,
        split="test",
    )

    print("\n=== Final Metrics ===")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall   : {metrics.box.mr:.4f}")
    print("\nPer-class results:")
    for row in metrics.summary():
        s = ''
        for key in row:
            s += f"{key}:{row[key]} "
        print(s)