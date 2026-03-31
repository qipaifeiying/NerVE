"""批量将 PKL 数据转换为点云文件（XYZ/PLY）。

输入目录结构示例：
    input_root/A/pred_nerve_pwl_curve.pkl

输出目录结构示例：
    output_root/A.xyz

参数直接在代码中配置，不依赖命令行参数。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


# ===== 在这里修改输入/输出路径参数 =====
INPUT_ROOT = Path(r"F:\skjWorkSpace\SourceCode\对比实验数据\NerVEdata")
OUTPUT_ROOT = Path(r"F:\skjWorkSpace\SourceCode\对比实验数据\NerVEdata_xyzverted")

# 输出格式："xyz" 或 "ply"
# 按你的示例 outpath/A.xyz，默认使用 xyz
OUTPUT_FORMAT = "xyz"

# 在每个子目录下寻找该文件名
INPUT_FILENAME = "pred_nerve_pwl_curve.pkl"


def _as_xyz_array(candidate: Any) -> np.ndarray | None:
    """尝试将对象解析为 Nx3 的浮点坐标数组，失败返回 None。"""
    try:
        arr = np.asarray(candidate)
    except Exception:
        return None

    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    if arr.shape[0] == 0:
        return None

    return arr[:, :3].astype(np.float32, copy=False)


def extract_points_from_pkl(obj: Any) -> np.ndarray:
    """从常见 PKL 结构中提取点云坐标。

    支持：
    1) 直接是 ndarray/list: Nx3(+)
    2) dict 中包含 points/xyz/vertices/coords/keypoints 等字段
    3) 嵌套 dict/list/tuple 中的上述结构
    """
    direct = _as_xyz_array(obj)
    if direct is not None:
        return direct

    preferred_keys = ("points", "xyz", "vertices", "coords", "keypoints", "pcd")

    def _dfs(node: Any) -> np.ndarray | None:
        arr = _as_xyz_array(node)
        if arr is not None:
            return arr

        if isinstance(node, dict):
            # 优先找常见键
            for key in preferred_keys:
                if key in node:
                    hit = _dfs(node[key])
                    if hit is not None:
                        return hit
            # 再遍历其余值
            for value in node.values():
                hit = _dfs(value)
                if hit is not None:
                    return hit

        elif isinstance(node, (list, tuple)):
            for item in node:
                hit = _dfs(item)
                if hit is not None:
                    return hit

        return None

    points = _dfs(obj)
    if points is None:
        raise ValueError("未能从 PKL 中解析出 Nx3 点云数据，请检查数据结构。")

    return points


def write_ascii_ply(output_path: Path, points_xyz: np.ndarray) -> None:
    """将点云写为 ASCII PLY（仅 x y z 三列）。"""
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points_xyz)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="\n") as f:
        f.write(header)
        # 使用固定小数位，兼顾可读性与体积
        np.savetxt(f, points_xyz, fmt="%.6f %.6f %.6f")


def write_ascii_xyz(output_path: Path, points_xyz: np.ndarray) -> None:
    """将点云写为 ASCII XYZ（仅 x y z 三列，无 header）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="\n") as f:
        np.savetxt(f, points_xyz, fmt="%.6f %.6f %.6f")


def convert_one(input_path: Path, output_path: Path, output_format: str) -> int:
    with input_path.open("rb") as f:
        obj = pickle.load(f)

    points_xyz = extract_points_from_pkl(obj)
    fmt = output_format.lower().strip(".")
    if fmt == "ply":
        write_ascii_ply(output_path, points_xyz)
    elif fmt == "xyz":
        write_ascii_xyz(output_path, points_xyz)
    else:
        raise ValueError(f"不支持的输出格式: {output_format}，仅支持 xyz/ply")

    return len(points_xyz)


def find_input_pkls(input_root: Path, filename: str) -> list[Path]:
    """查找 input_root 下一级子目录中的指定 pkl 文件。"""
    return sorted(input_root.glob(f"*/{filename}"))


def main() -> None:
    input_root = INPUT_ROOT
    output_root = OUTPUT_ROOT
    output_format = OUTPUT_FORMAT.lower().strip(".")

    if not input_root.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_root}")

    if output_format not in {"xyz", "ply"}:
        raise ValueError(f"OUTPUT_FORMAT 仅支持 xyz/ply，当前: {OUTPUT_FORMAT}")

    input_files = find_input_pkls(input_root, INPUT_FILENAME)
    if not input_files:
        raise FileNotFoundError(
            f"未找到输入文件，期望结构为: {input_root}/<name>/{INPUT_FILENAME}"
        )

    output_root.mkdir(parents=True, exist_ok=True)

    total_points = 0
    success = 0
    failed: list[tuple[Path, Exception]] = []

    for input_path in input_files:
        # input_root/A/pred_nerve_pwl_curve.pkl -> output_root/A.xyz(or .ply)
        sample_name = input_path.parent.name
        output_path = output_root / f"{sample_name}.{output_format}"

        try:
            count = convert_one(input_path, output_path, output_format)
            total_points += count
            success += 1
            print(f"[OK] {input_path} -> {output_path}，点数: {count}")
        except Exception as e:  # noqa: BLE001
            failed.append((input_path, e))
            print(f"[FAILED] {input_path}，原因: {e}")

    print("-" * 60)
    print(
        f"批处理完成: 成功 {success}/{len(input_files)}，"
        f"失败 {len(failed)}，总点数 {total_points}"
    )

    if failed:
        raise RuntimeError(
            "以下文件转换失败:\n"
            + "\n".join(f"- {path}: {err}" for path, err in failed)
        )


if __name__ == "__main__":
    main()
