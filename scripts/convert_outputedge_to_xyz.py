"""将输入目录中的 *_outputedge.ply 转换为 .xyz 文本点云。

参数直接在代码里配置，不依赖命令行参数。

转换规则：
    A_outputedge.ply -> <OUTPUT_DIR>/A.xyz
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


# ===== 在这里修改输入/输出路径参数 =====
INPUT_DIR = Path(r"F:\skjWorkSpace\SourceCode\对比实验数据\ECNet_result")
OUTPUT_DIR = Path(r"F:\skjWorkSpace\SourceCode\对比实验数据\ECNet_result_xyz")


def load_vertices_xyz(ply_path: Path) -> np.ndarray:
    """读取 PLY（点云或网格）并返回 Nx3 的 float32 顶点坐标。"""
    pcd = o3d.io.read_point_cloud(str(ply_path))
    vertices = np.asarray(pcd.points)

    # 某些 PLY 作为 mesh 才能正确读取
    if vertices.size == 0:
        mesh = o3d.io.read_triangle_mesh(str(ply_path))
        vertices = np.asarray(mesh.vertices)

    if vertices.ndim != 2 or vertices.shape[1] < 3:
        raise ValueError(f"顶点坐标维度不合法: {vertices.shape}, 文件: {ply_path}")

    return vertices[:, :3].astype(np.float32, copy=False)


def convert_one_file(ply_path: Path, out_dir: Path) -> tuple[Path, int]:
    """将单个 *_outputedge.ply 转为 .xyz，返回输出文件路径和点数。"""
    stem = ply_path.stem
    if not stem.endswith("_outputedge"):
        raise ValueError(f"文件名不符合规则（需以 _outputedge 结尾）: {ply_path.name}")

    base_name = stem[: -len("_outputedge")]
    out_path = out_dir / f"{base_name}.xyz"

    xyz = load_vertices_xyz(ply_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, xyz, fmt="%.6f", delimiter=" ")

    return out_path, len(xyz)


def main() -> None:
    input_dir = INPUT_DIR
    out_dir = OUTPUT_DIR

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在或不是目录: {input_dir}")

    ply_files = sorted(input_dir.glob("*_outputedge.ply"))
    if not ply_files:
        raise FileNotFoundError(f"未找到 *_outputedge.ply 文件: {input_dir}")

    total_points = 0
    converted = 0

    for ply_path in ply_files:
        out_path, point_count = convert_one_file(ply_path, out_dir)
        total_points += point_count
        converted += 1
        print(f"转换完成: {ply_path.name} -> {out_path.name}，点数: {point_count}")

    print(f"全部完成：{converted} 个文件，总点数: {total_points}，输出目录: {out_dir}")


if __name__ == "__main__":
    main()

