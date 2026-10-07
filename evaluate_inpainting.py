import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity


class MuralInpaintingEvaluator:
    """
    输入：
        RGB 图像：H x W x 3，范围 [0, 1]
        二值掩膜：H x W 或 H x W x 1，1 表示缺损区域

    默认直接评估最终合成图。
    compose=True 时，显式应用合成操作。
    """

    BINS = (
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.5),
        (0.5, 0.6),
    )

    def __init__(self, device=None, compose=False):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.compose = compose
        self.lpips_model = lpips.LPIPS(
            net="alex", version="0.1"
        ).to(self.device).eval()

    @classmethod
    def find_bin(cls, rho):
        for low, high in cls.BINS:
            if low < rho <= high:
                return f"({int(low * 100)},{int(high * 100)}]%"
        raise ValueError(
            f"Actual mask fraction {rho:.8f} is outside (0.1, 0.6]."
        )

    def evaluate_sample(self, gt_img, pred_img, mask):
        gt = np.asarray(gt_img, dtype=np.float64)
        pred = np.asarray(pred_img, dtype=np.float64)
        mask = np.asarray(mask)

        if gt.shape != pred.shape:
            raise ValueError("Prediction/reference shapes differ.")
        if gt.ndim != 3 or gt.shape[-1] != 3:
            raise ValueError("Expected H x W x 3 RGB images.")

        if mask.shape == gt.shape[:2] + (1,):
            mask = mask[..., 0]
        if mask.shape != gt.shape[:2]:
            raise ValueError("Mask/image shapes differ.")
        if not np.isin(mask, [0, 1]).all():
            raise ValueError("Mask must be binary; 1 means missing.")

        for name, img in (("reference", gt), ("prediction", pred)):
            if (
                not np.isfinite(img).all()
                or np.any(img < 0)
                or np.any(img > 1)
            ):
                raise ValueError(f"{name} must be finite and in [0, 1].")

        hole = mask.astype(bool)
        if not hole.any() or hole.all():
            raise ValueError("Both missing and observed pixels are required.")

        rho = float(hole.mean())
        bin_name = self.find_bin(rho)

        raw_outside_error = float(
            np.max(np.abs(pred[~hole] - gt[~hole]))
        )

        if self.compose:
            # Valid for the stated synthetic-damage benchmark:
            # observed pixels equal reference pixels outside the hole.
            evaluated = np.where(hole[..., None], pred, gt)
        else:
            # Do not silently overwrite a supposedly final output.
            if not np.array_equal(pred[~hole], gt[~hole]):
                raise ValueError(
                    "Observed pixels differ from the reference. "
                    "Supply the actual final composite, or use --compose "
                    "if the predictions are raw network outputs. "
                    f"Maximum outside error: {raw_outside_error:.8g}"
                )
            evaluated = pred

        squared_error = np.square(evaluated - gt)
        mse_full = float(squared_error.mean())
        mse_hole = float(squared_error[hole].mean())

        if not np.isclose(
            mse_full, rho * mse_hole, rtol=1e-12, atol=0.0
        ):
            raise ValueError("Full/hole MSE identity failed.")

        psnr_full = (
            np.inf if mse_full == 0 else -10.0 * np.log10(mse_full)
        )
        psnr_hole = (
            np.inf if mse_hole == 0 else -10.0 * np.log10(mse_hole)
        )
        expected_gap = -10.0 * np.log10(rho)

        if mse_full == 0:
            # Both PSNRs are infinite; inf - inf is undefined.
            observed_gap = np.nan
        else:
            observed_gap = psnr_full - psnr_hole
            if not np.isclose(
                observed_gap, expected_gap, rtol=0, atol=1e-10
            ):
                raise ValueError("Full/hole PSNR identity failed.")

        # Explicitly preserve the original skimage-default-style SSIM
        # configuration. Use the same configuration for every method.
        ssim_value = structural_similarity(
            gt,
            evaluated,
            data_range=1.0,
            channel_axis=-1,
            win_size=7,
            gaussian_weights=False,
            use_sample_covariance=True,
            K1=0.01,
            K2=0.03,
        )

        def to_lpips_tensor(img):
            array = np.ascontiguousarray(
                img.transpose(2, 0, 1), dtype=np.float32
            )
            return (
                torch.from_numpy(array)
                .unsqueeze(0)
                .to(self.device)
                * 2.0
                - 1.0
            )

        with torch.inference_mode():
            lpips_value = self.lpips_model(
                to_lpips_tensor(gt),
                to_lpips_tensor(evaluated),
            ).item()

        return {
            "mask_bin": bin_name,
            "mask_fraction": rho,
            "full_mse": mse_full,
            "hole_mse": mse_hole,
            "full_psnr": float(psnr_full),
            "hole_psnr": float(psnr_hole),
            "ssim": float(ssim_value),
            "lpips": float(lpips_value),
            "expected_gap_db": float(expected_gap),
            "observed_gap_db": float(observed_gap),
            "raw_outside_max_error": raw_outside_error,
            "composition_applied": self.compose,
        }


def read_rgb(path):
    """Read a lossless uint8 PNG without resizing or clipping."""
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError(f"Use lossless PNG images: {path}")

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected uint8 three-channel PNG: {path}")

    # OpenCV reads BGR; SSIM/LPIPS here use RGB.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float64) / 255.0


def read_mask(path, missing_value):
    """Explicitly interpret 0/255 or 0/1 binary masks."""
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError(f"Use a lossless PNG mask: {path}")

    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.dtype != np.uint8 or mask.ndim != 2:
        raise ValueError(f"Expected uint8 single-channel mask: {path}")

    allowed = {0, 1} if missing_value == 1 else {0, 255}
    if not set(np.unique(mask).tolist()).issubset(allowed):
        raise ValueError(f"Invalid binary encoding in mask: {path}")

    return mask == missing_value


def array_hash(array):
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()
    ).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError("No results to save.")

    with Path(path).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, expected_per_bin=None):
    groups = defaultdict(list)

    for row in rows:
        key = (row["method"], row["seed"])
        groups[(*key, row["mask_bin"])].append(row)
        groups[(*key, "all")].append(row)

    if expected_per_bin is not None:
        expected_names = [
            f"({int(a * 100)},{int(b * 100)}]%"
            for a, b in MuralInpaintingEvaluator.BINS
        ]
        for method, seed in {
            (row["method"], row["seed"]) for row in rows
        }:
            for bin_name in expected_names:
                count = len(groups.get((method, seed, bin_name), []))
                if count != expected_per_bin:
                    raise ValueError(
                        f"{method}, seed={seed}, {bin_name}: "
                        f"expected {expected_per_bin} samples, got {count}."
                    )

    summaries = []
    for (method, seed, bin_name), group in sorted(groups.items()):
        # Never silently discard infinite PSNR from zero-error samples.
        zero_count = sum(row["full_mse"] == 0 for row in group)

        full_mean = float(np.mean([r["full_psnr"] for r in group]))
        hole_mean = float(np.mean([r["hole_psnr"] for r in group]))
        expected_mean = float(
            np.mean([r["expected_gap_db"] for r in group])
        )
        observed_mean = (
            np.nan if zero_count else full_mean - hole_mean
        )

        if not zero_count and not np.isclose(
            observed_mean, expected_mean, rtol=0, atol=1e-10
        ):
            raise ValueError("Aggregate PSNR consistency check failed.")

        summaries.append({
            "method": method,
            "seed": seed,
            "mask_bin": bin_name,
            "n_pairs": len(group),
            "n_images": len({r["image_id"] for r in group}),
            "n_source_ids": len({r["source_id"] for r in group}),
            "zero_error_samples": zero_count,
            "full_psnr": full_mean,
            "hole_psnr": hole_mean,
            "ssim": float(np.mean([r["ssim"] for r in group])),
            "lpips": float(np.mean([r["lpips"] for r in group])),
            "mean_mask_fraction": float(
                np.mean([r["mask_fraction"] for r in group])
            ),
            "expected_mean_gap_db": expected_mean,
            "observed_mean_gap_db": observed_mean,
        })

    return summaries


def evaluate_manifest(args):
    manifest = args.manifest.resolve()
    required = {
        "method", "seed", "image_id", "source_id", "mask_id",
        "prediction", "target", "mask",
    }

    with manifest.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "Missing manifest columns: "
                + ", ".join(sorted(required - set(reader.fieldnames or [])))
            )
        records = list(reader)

    if not records:
        raise ValueError("The manifest is empty.")

    evaluator = MuralInpaintingEvaluator(
        device=args.device, compose=args.compose
    )
    rows = []
    seen = set()
    pair_signatures = {}
    source_map = {}
    case_sets = defaultdict(set)

    for record in records:
        if any(not (record.get(k) or "").strip() for k in required):
            raise ValueError("All required manifest fields must be nonempty.")

        method, seed = record["method"], record["seed"]
        image_id, mask_id = record["image_id"], record["mask_id"]
        source_id = record["source_id"]

        unique_key = (method, seed, image_id, mask_id)
        if unique_key in seen:
            raise ValueError(f"Duplicate evaluation record: {unique_key}")
        seen.add(unique_key)

        if image_id in source_map and source_map[image_id] != source_id:
            raise ValueError(f"Inconsistent source ID for {image_id}")
        source_map[image_id] = source_id

        paths = {
            name: (manifest.parent / record[name]).resolve()
            for name in ("prediction", "target", "mask")
        }
        gt = read_rgb(paths["target"])
        pred = read_rgb(paths["prediction"])
        mask = read_mask(paths["mask"], args.missing_value)

        # Check that each image-mask case is identical across methods/seeds.
        case = (image_id, mask_id)
        signature = (gt.shape, array_hash(gt), array_hash(mask))
        if case in pair_signatures and pair_signatures[case] != signature:
            raise ValueError(
                f"Reference or mask differs across methods/seeds: {case}"
            )
        pair_signatures[case] = signature
        case_sets[(method, seed)].add(case)

        metrics = evaluator.evaluate_sample(gt, pred, mask)
        rows.append({
            "method": method,
            "seed": seed,
            "image_id": image_id,
            "source_id": source_id,
            "mask_id": mask_id,
            **metrics,
        })

    # All evaluated method/seed groups must use the same cases.
    reference_cases = next(iter(case_sets.values()))
    for key, cases in case_sets.items():
        if cases != reference_cases:
            raise ValueError(f"Evaluation cases differ for {key}.")

    summary = summarize(rows, args.expected_per_bin)

    # Prevent accidental replacement of previous evaluation records.
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "per_sample.csv", rows)
    write_csv(args.output / "per_bin.csv", summary)

    for row in summary:
        print(
            f"{row['method']} | seed={row['seed']} | "
            f"{row['mask_bin']} | N={row['n_pairs']} | "
            f"Full={row['full_psnr']:.4f} | "
            f"Hole={row['hole_psnr']:.4f} | "
            f"SSIM={row['ssim']:.6f} | "
            f"LPIPS={row['lpips']:.6f}"
        )

    print(f"\nSaved results to: {args.output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate mural inpainting from real prediction files."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--missing-value",
        type=int,
        choices=[0, 1, 255],
        default=255,
        help="Mask value denoting missing pixels.",
    )
    parser.add_argument(
        "--compose",
        action="store_true",
        help="Explicitly composite raw predictions using observed pixels.",
    )
    parser.add_argument(
        "--expected-per-bin",
        type=int,
        default=None,
        help="Require this many cases per bin for each method/seed.",
    )
    evaluate_manifest(parser.parse_args())
