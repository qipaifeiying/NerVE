"""
基于 Open3D 的体素下采样脚本（PLY / XYZ）。

使用方式：
1) 在代码中的 CONFIG 里设置输入文件夹路径、输出文件夹路径、voxel_size。
2) 运行：python scripts/VoxelSample.py

说明：
- 本脚本会批量处理输入文件夹中的 .ply / .xyz 点云。
- 输出文件名会保留原名并追加后缀 _voxel_downsample。
- 体素下采样核心接口为 open3d.geometry.PointCloud.voxel_down_sample。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import open3d as o3d


@dataclass
class VoxelSampleConfig:
    """体素下采样配置。"""

    # 输入/输出文件夹路径
    input_dir: str
    output_dir: str

    # 核心参数：体素尺寸（单位与点云坐标单位一致）
    voxel_size: float = 0.001

    # I/O 常用控制参数
    write_ascii: bool = True
    compressed: bool = False
    print_progress: bool = True

    # 可选控制：是否在下采样后重新估计法向（若原始点云有法向，可按需保留或重估）
    reestimate_normals: bool = False
    normal_search_radius: float = 0.03
    normal_max_nn: int = 30


CONFIG = VoxelSampleConfig(
    input_dir=r"F:\skjWorkSpace\SourceCode\对比实验数据\ECNet_result_xyz",
    output_dir=r"F:\skjWorkSpace\SourceCode\对比实验数据\ECNet_results",
    voxel_size=0.02,
    write_ascii=True,
    compressed=False,
    print_progress=True,
    reestimate_normals=False,
    normal_search_radius=0.03,
    normal_max_nn=30,
)


def _ensure_supported_point_cloud(path: Path, field_name: str) -> None:
    supported_suffixes = {".ply", ".xyz"}
    suffix = path.suffix.lower()
    if suffix not in supported_suffixes:
        allowed = ", ".join(sorted(supported_suffixes))
        raise ValueError(f"{field_name} 必须为以下格式之一: {allowed}，当前文件: {path}")


def _validate_config(cfg: VoxelSampleConfig) -> tuple[Path, Path]:
    in_dir = Path(cfg.input_dir)
    out_dir = Path(cfg.output_dir)

    if not in_dir.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {in_dir}")
    if not in_dir.is_dir():
        raise ValueError(f"input_dir 必须为文件夹路径: {in_dir}")

    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"output_dir 必须为文件夹路径: {out_dir}")

    if cfg.voxel_size <= 0:
        raise ValueError(f"voxel_size 必须 > 0，当前值: {cfg.voxel_size}")

    if cfg.reestimate_normals:
        if cfg.normal_search_radius <= 0:
            raise ValueError(
                f"normal_search_radius 必须 > 0，当前值: {cfg.normal_search_radius}"
            )
        if cfg.normal_max_nn <= 0:
            raise ValueError(f"normal_max_nn 必须 > 0，当前值: {cfg.normal_max_nn}")

    return in_dir, out_dir


def _process_one_file(in_path: Path, out_path: Path, cfg: VoxelSampleConfig) -> None:
    print(f"[Info] 读取点云: {in_path}")
    pcd = o3d.io.read_point_cloud(str(in_path), print_progress=cfg.print_progress)

    if pcd.is_empty():
        raise RuntimeError(f"输入点云为空或读取失败: {in_path}")

    n_in = len(pcd.points)
    print(f"[Info] 输入点数: {n_in}")

    # 给出可调参数提示：根据包围盒对角线自动建议 voxel_size 量级。
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    diag = float(np.linalg.norm(extent))
    print(f"[Hint] 点云包围盒尺寸(Ex,Ey,Ez)=({extent[0]:.6g}, {extent[1]:.6g}, {extent[2]:.6g})")
    print(f"[Hint] 包围盒对角线长度={diag:.6g}")
    if diag > 0:
        s1 = diag * 0.001
        s2 = diag * 0.003
        s3 = diag * 0.01
        print(
            "[Hint] 当前保留比例高时，请优先调大 voxel_size。"
            f"建议先尝试: {s1:.6g}, {s2:.6g}, {s3:.6g}"
        )
    else:
        print("[Hint] 点云尺度异常（包围盒对角线为0），请检查输入数据。")

    print("[Hint] 主要调参项: voxel_size（越大降采样越强）")
    print("[Hint] 非采样强度参数: write_ascii / compressed / print_progress")

    print(f"[Info] 开始体素下采样, voxel_size={cfg.voxel_size}")

    pcd_down = pcd.voxel_down_sample(voxel_size=cfg.voxel_size)

    if cfg.reestimate_normals:
        print("[Info] 重新估计法向")
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=cfg.normal_search_radius,
                max_nn=cfg.normal_max_nn,
            )
        )

    n_out = len(pcd_down.points)
    ratio = n_out / n_in if n_in > 0 else 0.0
    print(f"[Info] 输出点数: {n_out}")
    print(f"[Info] 保留比例: {ratio:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(
        str(out_path),
        pcd_down,
        write_ascii=cfg.write_ascii,
        compressed=cfg.compressed,
        print_progress=cfg.print_progress,
    )

    if not ok:
        raise RuntimeError(f"输出点云写入失败: {out_path}")

    print(f"[Info] 已保存下采样点云: {out_path}")


def run_voxel_sampling(cfg: VoxelSampleConfig) -> None:
    in_dir, out_dir = _validate_config(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        p
        for p in in_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".ply", ".xyz"}
    )

    if not candidates:
        raise RuntimeError(f"输入文件夹中未找到 .ply 或 .xyz 文件: {in_dir}")

    print(f"[Info] 输入文件夹: {in_dir}")
    print(f"[Info] 输出文件夹: {out_dir}")
    print(f"[Info] 待处理文件数: {len(candidates)}")

    success = 0
    failed = 0
    for src in candidates:
        _ensure_supported_point_cloud(src, "input_file")
        dst = out_dir / f"{src.stem}_voxel_downsample{src.suffix.lower()}"
        print("-" * 72)
        try:
            _process_one_file(src, dst, cfg)
            success += 1
        except Exception as exc:
            failed += 1
            print(f"[Error] 处理失败: {src}，原因: {exc}")

    print("=" * 72)
    print(f"[Info] 批处理完成：成功 {success}，失败 {failed}，总计 {len(candidates)}")


if __name__ == "__main__":
    try:
        run_voxel_sampling(CONFIG)
    except Exception as exc:
        print(f"[Error] {exc}")
        sys.exit(1)
