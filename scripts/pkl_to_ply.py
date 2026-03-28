"""将 PKL 数据转换为 ASCII PLY 点云（仅 XYZ）。

参数直接在代码中配置，不依赖命令行参数。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


# ===== 在这里修改输入/输出路径参数 =====
INPUT_PKL_PATH = Path(r"F:\skjWorkSpace\SourceCode\NerVE\demo\03\cad_pwl_curve.pkl")
OUTPUT_PLY_PATH = Path(r"F:\skjWorkSpace\SourceCode\NerVE\demo\03\cad_pwl_curve.ply")


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


def convert_pkl_to_ascii_ply(input_path: Path, output_path: Path) -> int:
    with input_path.open("rb") as f:
        obj = pickle.load(f)

    points_xyz = extract_points_from_pkl(obj)
    write_ascii_ply(output_path, points_xyz)
    return len(points_xyz)


def main() -> None:
    input_path = INPUT_PKL_PATH
    output_path = OUTPUT_PLY_PATH

    if not input_path.exists():
        raise FileNotFoundError(f"输入 PKL 文件不存在: {input_path}")

    count = convert_pkl_to_ascii_ply(input_path, output_path)
    print(f"转换完成: {input_path} -> {output_path}，点数: {count}")


if __name__ == "__main__":
    main()
