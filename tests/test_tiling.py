import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import starpack


class TilingSeedTests(unittest.TestCase):
  def test_interlocking_fifty_star_seed(self):
    aspect = 100 / 71
    base = starpack.make_base_star(5, 2)
    state, height = starpack.regular_tiling_seed(50, aspect, base, "{5/2}")
    self.assertEqual(state.shape, (50, 3))
    self.assertEqual(np.count_nonzero(state[:, 2] == 0), 25)
    self.assertTrue(starpack.exact_valid(state, height, aspect, base))
    self.assertAlmostEqual(starpack.tight_box(state, aspect, base)[1], height)
    self.assertLess(aspect * height * height, 105)

  def test_other_symbols_have_valid_regular_seed(self):
    base = starpack.make_base_star(7, 2)
    state, height = starpack.regular_tiling_seed(12, 1.4, base, "{7/2}")
    self.assertTrue(starpack.exact_valid(state, height, 1.4, base))

  def test_no_load_saves_seed_before_search(self):
    with tempfile.TemporaryDirectory() as directory:
      command = [sys.executable, str(Path(starpack.__file__).resolve()), "{7/2}", "4",
                 "--aspect", "1.4", "--epochs", "1", "--workers", "1", "--iterations", "1",
                 "--contact-steps", "0", "--snap-rounds", "0", "--jostle-every", "0",
                 "--image-width", "140"]
      run = subprocess.run(command, cwd=directory, check=True, capture_output=True, text=True, timeout=60)
      self.assertIn("regular tiling seed:", run.stdout)
      result = Path(directory, "results", "7-2_4.json")
      with open(result) as stream:
        saved = json.load(stream)
      state = np.array([[star[key] for key in ("x", "y", "theta")] for star in saved["packing"]])
      self.assertTrue(starpack.exact_valid(state, saved["height"], 1.4, starpack.make_base_star(7, 2)))
      self.assertTrue(result.with_suffix(".png").exists())


if __name__ == "__main__":
  unittest.main()
