"""SPEC 17 item 16: the committed smoke assets must be usable test material."""

from __future__ import annotations

from PIL import Image

CONTROL = "vast_comfy_smoke_control.png"
INPAINT = "vast_comfy_smoke_inpaint_src.png"


def test_control_asset_is_a_high_contrast_white_lineart_image(repo_root):
    path = repo_root / "assets" / "smoke" / CONTROL
    assert path.is_file()
    with Image.open(path) as img:
        img.load()
        assert img.size == (1024, 1024)
        rgb = img.convert("RGB")
        pixels = list(rgb.get_flattened_data()) if hasattr(rgb, "get_flattened_data") else list(rgb.getdata())

    # Mostly white (it is paper with thin strokes) ...
    white = sum(1 for r, g, b in pixels if r >= 230 and g >= 230 and b >= 230)
    assert white / len(pixels) > 0.5
    # ... but high contrast: real dark ink must exist.
    dark = sum(1 for r, g, b in pixels if r < 80 and g < 80 and b < 80)
    assert dark / len(pixels) > 0.001
    # Very small file (a management asset, not user data).
    assert path.stat().st_size < 1_000_000


def test_inpaint_asset_has_a_transparent_region_in_the_middle(repo_root):
    path = repo_root / "assets" / "smoke" / INPAINT
    assert path.is_file()
    with Image.open(path) as img:
        img.load()
        assert img.size == (1024, 1024)
        assert img.mode in ("RGBA", "LA")
        alpha = img.getchannel("A")
        assert alpha.getextrema() == (0, 255)  # fully transparent pixels exist

        width, height = img.size
        middle = alpha.crop((width // 4, height // 4, 3 * width // 4, 3 * height // 4))
        middle_pixels = list(middle.get_flattened_data()) if hasattr(middle, "get_flattened_data") else list(middle.getdata())
        transparent = sum(1 for value in middle_pixels if value == 0)
        assert transparent > 100, "the center of the inpaint source must be fully transparent"

        # The border area must be opaque so the mask is not the whole image.
        corners = alpha.crop((0, 0, 64, 64))
        corner_pixels = list(corners.get_flattened_data()) if hasattr(corners, "get_flattened_data") else list(corners.getdata())
        assert all(value == 255 for value in corner_pixels)
    assert path.stat().st_size < 1_000_000


def test_smoke_assets_are_whitelisted_for_upload():
    from vast_comfy.remote import package_files

    rels = {rel for _, rel in package_files()}
    assert f"assets/smoke/{CONTROL}" in rels
    assert f"assets/smoke/{INPAINT}" in rels
