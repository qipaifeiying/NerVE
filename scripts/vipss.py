"""批量调用 vipss 可执行文件处理输入目录中的文件。

参数直接在代码中配置，不依赖命令行参数。

对输入目录下每个文件执行：
    vipss -i <inputfile> -o <outpath> -s
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


# ===== 在这里修改批处理参数 =====
VIPSS_EXE = Path(r"F:\tools\vipss.exe")
INPUT_PATH = Path(r"F:\skjWorkSpace\SourceCode\NerVE\NerVE_data_xyz")
OUTPUT_ROOT = Path(r"F:\skjWorkSpace\SourceCode\NerVE\vipss_out")
FILE_PATTERN = "*.xyz"


def resolve_config_path(path_value: Path) -> Path:
    """将配置路径解析为绝对路径：相对路径按脚本所在目录解析。"""
    if path_value.is_absolute():
        return path_value
    base_dir = Path(__file__).resolve().parent
    return (base_dir / path_value).resolve()


def run_one(vipss_exe: Path, input_file: Path, output_dir: Path) -> None:
    """执行单个文件的 vipss 命令。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    # vipss 在部分版本中会把 -o 当作“前缀字符串”拼接，
    # 这里强制补一个路径分隔符，避免出现：NerVEdata_results00000003_normal.ply
    output_arg = str(output_dir.resolve()) + os.sep

    cmd = [
        str(vipss_exe),
        "-i",
        str(input_file),
        "-o",
        output_arg,
        "-s 100",
    ]

    subprocess.run(cmd, check=True)


def main() -> None:
    vipss_exe = resolve_config_path(VIPSS_EXE)
    input_dir = resolve_config_path(INPUT_PATH)
    output_root = resolve_config_path(OUTPUT_ROOT)

    if not vipss_exe.exists():
        raise FileNotFoundError(f"vipss 可执行文件不存在: {vipss_exe}")

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在或不是目录: {input_dir}")

    input_files = sorted(p for p in input_dir.glob(FILE_PATTERN) if p.is_file())
    if not input_files:
        raise FileNotFoundError(f"输入目录下未找到匹配文件（{FILE_PATTERN}）: {input_dir}")

    # 注意：vipss 的 -o 参数通常作为“输出前缀/输出路径基名”使用，
    # 不是“每个输入对应一个输出目录”。
    # 若传入 output_root / input_file.stem，会出现类似
    # 00000003 + 00000003_normal.ply -> 0000000300000003_normal.ply 的重复命名。
    output_root.mkdir(parents=True, exist_ok=True)

    success = 0
    for input_file in input_files:
        print(f"开始处理: {input_file} -> {output_root}")
        run_one(vipss_exe, input_file, output_root)
        success += 1
        print(f"处理完成: {input_file.name}")

    print(f"批处理结束，共成功处理 {success} 个文件。输出根目录: {output_root}")


if __name__ == "__main__":
    main()
