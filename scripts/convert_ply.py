"""批量将输入目录中的 PLY 转换为仅包含 XYZ 的 binary_little_endian PLY。

参数直接在代码里配置，不依赖命令行参数。

输出结构：
- 输入: A.ply
- 输出: <OUTPUT_ROOT_DIR>/A/pc_obj.ply
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


# ===== 在这里修改输入/输出路径参数 =====
INPUT_DIR = Path(r"F:\skjWorkSpace\SourceCode\NerVE\NerVE_data")
OUTPUT_ROOT_DIR = Path(r"F:\skjWorkSpace\SourceCode\NerVE\NerVE_data_converted")


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
    if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"输入目录不存在或不是目录: {INPUT_DIR}")

    input_files = sorted(p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".ply")
    if not input_files:
        raise FileNotFoundError(f"输入目录下未找到 .ply 文件: {INPUT_DIR}")

    total_points = 0
    success_count = 0

    for input_path in input_files:
        output_path = OUTPUT_ROOT_DIR / input_path.stem / "pc_obj.ply"
        count = convert_ply_to_xyz_binary(input_path, output_path)
        total_points += count
        success_count += 1
        print(f"转换完成: {input_path.name} -> {output_path}，点数: {count}")

    print(
        f"批处理完成，共转换 {success_count} 个文件，总点数: {total_points}。"
        f"输出根目录: {OUTPUT_ROOT_DIR}"
    )


if __name__ == "__main__":
    main()
