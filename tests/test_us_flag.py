import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import starpack


class USFlagTests(unittest.TestCase):
  def test_flag_geometry_and_palette_override(self):
    base = starpack.make_base_star(5, 2)
    height = 3.0
    aspect = starpack.US_CANTON_ASPECT
    state = np.array([[height * aspect / 2, height / 2, 0.0]])
    with tempfile.TemporaryDirectory() as directory:
      filename = os.path.join(directory, "flag.png")
      starpack.render_packing_image(filename, state, height, aspect, base, image_width=1900,
                                    background="#123456", border="#654321", fill="#ABCDEF", us_flag=True)
      with Image.open(filename) as image:
        self.assertEqual(image.size, (1900, 1000))
        self.assertEqual(image.getpixel((1800, 1)), (178, 34, 52))
        self.assertEqual(image.getpixel((1800, 77)), (255, 255, 255))
        self.assertEqual(image.getpixel((1800, 999)), (178, 34, 52))
        self.assertEqual(image.getpixel((1, 1)), (60, 59, 110))
        self.assertEqual(image.getpixel((759, 1)), (60, 59, 110))
        self.assertEqual(image.getpixel((760, 1)), (178, 34, 52))
        self.assertEqual(image.getpixel((1, 537)), (60, 59, 110))
        self.assertEqual(image.getpixel((1, 538)), (255, 255, 255))
        self.assertEqual(image.getpixel((380, 269)), (255, 255, 255))
        self.assertEqual({colour for _, colour in image.getcolors(image.width * image.height)},
                         {(178, 34, 52), (60, 59, 110), (255, 255, 255)})

  def test_cli_converts_existing_packing_and_overrides_aspect(self):
    with tempfile.TemporaryDirectory() as directory:
      result_dir = Path(directory, "results")
      result_dir.mkdir()
      base = starpack.make_base_star(5, 2)
      state, height = starpack.tight_box(np.array([[1.5, 1.5, 0.0]]), 1.0, base)
      starpack.save_solution(str(result_dir / "5-2_1.json"), str(result_dir / "5-2_1.png"),
                             state, height, 1.0, "{5/2}", base, 0, 1, 100, "#FFFFFF", "#000000", "#808080")
      command = [sys.executable, str(Path(starpack.__file__).resolve()), "{5/2}", "1", "--us-flag",
                 "--aspect", "4", "--background", "#123456", "--fill", "#123456",
                 "--border", "#123456", "--image-width", "190", "--epochs", "1", "--workers", "1",
                 "--iterations", "1", "--contact-steps", "0", "--snap-rounds", "0", "--jostle-every", "0"]
      subprocess.run(command, cwd=directory, check=True, capture_output=True, text=True, timeout=60)
      with open(result_dir / "5-2_1_us_flag.json") as stream:
        output = json.load(stream)
      self.assertAlmostEqual(output["aspect_ratio"], starpack.US_CANTON_ASPECT)
      with Image.open(result_dir / "5-2_1_us_flag.png") as image:
        self.assertEqual(image.size, (190, 100))
        self.assertEqual(image.getpixel((180, 1)), (178, 34, 52))
      with open(result_dir / "5-2_1.json") as stream:
        self.assertEqual(json.load(stream)["aspect_ratio"], 1.0)


if __name__ == "__main__":
  unittest.main()
