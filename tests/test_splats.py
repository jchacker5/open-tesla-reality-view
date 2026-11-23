import numpy as np
import pytest
from splatfix import make_ply, path_dict

from tsplat import splats


def test_ply_round_trip_keeps_every_property_in_order(tmp_path):
    ply = make_ply(np.random.default_rng(0).normal(size=(50, 3)), opacity=0.3)
    splats.write_ply(tmp_path / "a.ply", ply)
    back = splats.read_ply(tmp_path / "a.ply")
    assert back.data.dtype.names == ply.data.dtype.names
    assert back.comments == ply.comments
    for n in ply.data.dtype.names:
        assert np.array_equal(back.data[n], ply.data[n])
    assert np.allclose(back.opacity(), 0.3)
    head = (tmp_path / "a.ply").read_bytes()[:200]
    assert b"binary_little_endian" in head and b"property float f_dc_0" in head


def test_read_ply_converts_other_scalar_types_to_float_on_write(tmp_path):
    data = np.zeros(3, dtype=[("x", "<f8"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("opacity", "<f4")])
    data["x"] = [1, 2, 3]
    data["red"] = [10, 20, 30]
    head = "ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
    head += "property double x\nproperty float y\nproperty float z\nproperty uchar red\nproperty float opacity\n"
    (tmp_path / "b.ply").write_bytes((head + "end_header\n").encode() + data.tobytes())
    ply = splats.read_ply(tmp_path / "b.ply")
    assert list(ply.data["red"]) == [10, 20, 30] and list(ply.data["x"]) == [1, 2, 3]
    splats.write_ply(tmp_path / "c.ply", ply)
    again = splats.read_ply(tmp_path / "c.ply")
    assert again.data.dtype["red"] == np.dtype("<f4") and list(again.data["red"]) == [10, 20, 30]


def test_read_ply_rejects_ascii(tmp_path):
    (tmp_path / "a.ply").write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n1\n")
    with pytest.raises(ValueError, match="binary_little_endian"):
        splats.read_ply(tmp_path / "a.ply")


@pytest.mark.parametrize("mpu", [1.0, 4.0])
def test_tube_mask_lateral_and_height_limits_in_metres(mpu):
    ego = splats.EgoPath.from_dict(path_dict(mpu=mpu))
    u = 1 / mpu  # one metre in scene units; road plane is y = 0, up = -y
    pts = np.array(
        [
            [0.0, -1.0 * u, 10],  # on the path, 1 m above the road: inside
            [0.9 * u, -0.5 * u, 10],  # 0.9 m to the side: inside
            [1.2 * u, -0.5 * u, 10],  # 1.2 m to the side: outside
            [0.0, -0.05 * u, 10],  # road surface (5 cm): outside, the road is not a floater
            [0.0, -1.8 * u, 10],  # 1.8 m up (above the roof): outside
            [0.0, -1.0 * u, 30],  # beyond the end of the path (z 0..20): outside
        ]
    )
    assert splats.EgoPath.from_dict(path_dict(mpu=mpu)).tube_mask(pts).tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    lateral, height, _ = ego.nearest_road(pts[:2])
    assert lateral == pytest.approx([0.0, 0.9], abs=0.06)
    assert height == pytest.approx([1.0, 0.5], abs=1e-6)


def test_floaters_counts_only_opaque_splats_in_the_tube():
    ego = splats.EgoPath.from_dict(path_dict())
    inside = [[0, -1.0, z] for z in range(5)]
    outside = [[5, -1.0, z] for z in range(5)]
    ply = make_ply(inside + outside + [[0, -1.0, 7]], opacity=[0.9] * 10 + [0.05])
    assert splats.floaters(ply, ego) == 5


def test_scales_and_colour_helpers():
    ply = make_ply([[0, 0, 0]], scale_m=2.0, mpu=1.0)
    ply.data["scale_1"] = np.log(0.1)
    assert ply.max_scale()[0] == pytest.approx(2.0) and ply.min_scale()[0] == pytest.approx(0.1)
    ply.data["f_dc_0"] = 0
    assert ply.rgb()[0, 0] == pytest.approx(0.5)


def test_horizontal_basis_is_orthonormal_and_follows_the_path():
    ego = splats.EgoPath.from_dict(path_dict())
    right, fwd = ego.horizontal_basis()
    assert np.allclose(fwd, [0, 0, 1]) and abs(right @ ego.up) < 1e-9 and abs(right @ fwd) < 1e-9
    assert np.allclose(np.cross(right, ego.up), -fwd)  # (right, up, -fwd) is right-handed
