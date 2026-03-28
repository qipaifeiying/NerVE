"""将 PLY 点云转换为 XYZ 文本格式文件。

参数直接在代码里配置，不依赖命令行参数。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


# ===== 在这里修改输入/输出路径参数 =====
INPUT_PLY_PATH = Path(r"F:\skjWorkSpace\SourceCode\CurveNetworkRecon\data\CADinput\00000003.ply")
OUTPUT_XYZ_PATH = Path(r"F:\skjWorkSpace\SourceCode\CurveNetworkRecon\data\CADinput\00000003.xyz")


def load_vertices(input_path: Path) -> np.ndarray:
    """读取 PLY（point cloud/mesh）并提取顶点坐标。"""
    pcd = o3d.io.read_point_cloud(str(input_path))
    vertices = np.asarray(pcd.points)

    # 有些 PLY 会被当作 mesh，点云读取为空时尝试 mesh 读取
    if vertices.size == 0:
        mesh = o3d.io.read_triangle_mesh(str(input_path))
        vertices = np.asarray(mesh.vertices)

    if vertices.ndim != 2 or vertices.shape[1] < 3:
        raise ValueError(f"顶点坐标维度不合法: {vertices.shape}")

    return vertices[:, :3].astype(np.float32, copy=False)


def write_xyz_text(output_path: Path, vertices_xyz: np.ndarray) -> None:
    """写出 XYZ 文本文件（每行: x y z）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, vertices_xyz, fmt="%.6f", delimiter=" ")


def convert_ply_to_xyz_text(input_path: Path, output_path: Path) -> int:
    vertices_xyz = load_vertices(input_path)
    write_xyz_text(output_path, vertices_xyz)
    return len(vertices_xyz)


def main() -> None:
    input_path = INPUT_PLY_PATH
    output_path = OUTPUT_XYZ_PATH

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    count = convert_ply_to_xyz_text(input_path, output_path)
    print(f"转换完成: {input_path} -> {output_path}，点数: {count}")


if __name__ == "__main__":
    main()
