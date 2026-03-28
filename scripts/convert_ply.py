"""将 PLY 点云转换为仅包含 XYZ 坐标的二进制 little-endian PLY。

参数直接在代码里配置，不依赖命令行参数。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


# ===== 在这里修改输入/输出路径参数 =====
INPUT_PLY_PATH = Path(r"F:\skjWorkSpace\SourceCode\CurveNetworkRecon\data\CADinput\00000003.ply")
OUTPUT_PLY_PATH = Path(r"F:\skjWorkSpace\SourceCode\CurveNetworkRecon\data\CADinput\00000003xyz_binary.ply")


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


def write_binary_xyz_ply(output_path: Path, vertices_xyz: np.ndarray) -> None:
    """写出只含 x/y/z(float) 的 binary_little_endian PLY。"""
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices_xyz)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertices_xyz.tobytes(order="C"))


def convert_ply_to_xyz_binary(input_path: Path, output_path: Path) -> int:
    vertices_xyz = load_vertices(input_path)
    write_binary_xyz_ply(output_path, vertices_xyz)
    return len(vertices_xyz)


def main() -> None:
    input_path = INPUT_PLY_PATH
    output_path = OUTPUT_PLY_PATH

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    count = convert_ply_to_xyz_binary(input_path, output_path)
    print(f"转换完成: {input_path} -> {output_path}，点数: {count}")


if __name__ == "__main__":
    main()
