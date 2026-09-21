import numpy as np

from morse_lidar.pointcloud import PointCloud, load_point_cloud, rasterize_depth, voxel_downsample


def test_ascii_ply_loads_and_rasterizes(tmp_path):
    source = tmp_path / "scan.ply"
    points = [(float(x), float(y), float(x + y)) for y in range(5) for x in range(6)]
    with source.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\nelement vertex 30\n")
        handle.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        handle.writelines(f"{x} {y} {z}\n" for x, y, z in points)
    cloud = load_point_cloud(source)
    raster = rasterize_depth(cloud, rows=5, cols=6)
    assert cloud.points.shape == (30, 3)
    assert raster.coverage == 1.0
    assert np.isfinite(raster.depth).all()


def test_voxel_downsample_reduces_duplicates():
    cloud = PointCloud(np.array([[0, 0, 0], [0.01, 0.01, 0.01], [1, 1, 1]], dtype=float))
    assert len(voxel_downsample(cloud, 0.1).points) == 2