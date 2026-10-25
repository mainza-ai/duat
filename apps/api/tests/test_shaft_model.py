"""Helical-ramp shaft model: sign conventions, ramp solidity, relief variant."""
import numpy as np

from apps.api.pipeline.shaft_model import ShaftParams, shaft_void_sdf

P = ShaftParams()
CX, CY = -40.0, -20.0


def test_axis_is_void_wall_is_solid():
    axis = shaft_void_sdf(np.array([CX]), np.array([CY]), np.array([-100.0]), CX, CY, P)
    assert float(axis[0]) < 0.0
    wall = shaft_void_sdf(np.array([CX + 50.0]), np.array([CY]), np.array([-100.0]), CX, CY, P)
    assert float(wall[0]) > 0.0
    above = shaft_void_sdf(np.array([CX]), np.array([CY]), np.array([0.0]), CX, CY, P)
    assert float(above[0]) > 0.0


def test_ramp_shelf_is_solid_inside_void():
    # Ramp shelf spans r in [R-w, R+0.5]; sample its mid-radius at a ramp height.
    # Ramp centerline passes z_top at theta=0: pick theta=0, z=z_top.
    r_mid = P.radius_m - P.ramp_width_m / 2.0
    v = shaft_void_sdf(np.array([CX + r_mid]), np.array([CY]),
                        np.array([P.z_top_m]), CX, CY, P)
    assert float(v[0]) > 0.0, "ramp shelf must read solid (carved out of the void)"
    # Same radius half a pitch above/below the turn: open void.
    v2 = shaft_void_sdf(np.array([CX + r_mid]), np.array([CY]),
                        np.array([P.z_top_m - P.spiral_pitch_m / 2.0]), CX, CY, P)
    assert float(v2[0]) < 0.0


def test_plain_cylinder_differs_from_relief_variant():
    xs = np.linspace(CX - 8, CX + 8, 40)
    X, Y = np.meshgrid(xs, xs)
    Z = np.full_like(X, -100.0)
    a = shaft_void_sdf(X, Y, Z, CX, CY, P)
    b = shaft_void_sdf(X, Y, Z, CX, CY,
                       ShaftParams(wall_relief_m=1.1, wall_relief_lobes=2))
    assert not np.allclose(a, b)
    # Plain variant is a true cylinder: SDF depends only on radius here.
    r = np.sqrt((X - CX) ** 2 + (Y - CY) ** 2)
    ring = (np.abs(r - P.radius_m) < 0.05)
    assert np.all(np.abs(a[ring]) < 0.06)


def test_renderer_matches_volume_convention():
    from apps.api.pipeline.depth_renderer import render_shaft_planar_depth_and_reflectivity

    log_u8, _, valid = render_shaft_planar_depth_and_reflectivity(
        (CX + 1.0, CY, -45.0), width=160, height=90, z_min=0.5, z_max=40.0,
        noise_sigma_m=0.0, seed=7)
    # Enclosed shaft viewed from inside: wall + end caps cover the frame.
    assert float(valid.mean()) > 0.9
