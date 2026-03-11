import open3d as o3d
import os
import sys
import argparse
from pathlib import Path

def process_stl(input_path: Path, output_path: Path, target_faces: int):
    try:
        # 加载网格
        mesh = o3d.io.read_triangle_mesh(str(input_path))
        if not mesh.has_triangles():
            print(f"⚠️  Skipped {input_path.name}: no triangles")
            return False

        original_faces = len(mesh.triangles)
        print(f"📄 {input_path.name}: {original_faces} faces", end="")

        # 修复网格
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()

        # 简化（仅当超过目标面数）
        if len(mesh.triangles) > target_faces:
            mesh = mesh.simplify_quadric_decimation(target_faces)

        # 关键：计算法线（否则 STL 写入失败）
        mesh.compute_vertex_normals()

        # 导出为二进制 STL
        success = o3d.io.write_triangle_mesh(
            str(output_path),
            mesh,
            write_ascii=False,
            compressed=False
        )

        if success:
            final_faces = len(mesh.triangles)
            print(f" → {final_faces} faces ✅")
            return True
        else:
            print(f" ❌ Write failed!")
            return False

    except Exception as e:
        print(f" ❌ Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch fix and simplify STL files for MuJoCo.")
    parser.add_argument("input_dir", help="Input directory containing .stl files")
    parser.add_argument("output_dir", help="Output directory for fixed .stl files")
    parser.add_argument("--target_faces", type=int, default=50000,
                        help="Target face count after simplification (default: 50000)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    target_faces = args.target_faces

    if not input_dir.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    stl_files = list(input_dir.glob("*.stl")) + list(input_dir.glob("*.STL"))
    if not stl_files:
        print(f"⚠️  No .stl files found in {input_dir}")
        return

    print(f"🔍 Found {len(stl_files)} STL files. Processing...\n")

    success_count = 0
    for stl_file in sorted(stl_files):
        out_file = output_dir / stl_file.name
        if process_stl(stl_file, out_file, target_faces):
            success_count += 1

    print(f"\n✅ Done! {success_count}/{len(stl_files)} files processed successfully.")
    print(f"📁 Output saved to: {output_dir}")

if __name__ == "__main__":
    main()