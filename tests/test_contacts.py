import math
import unittest

import numpy as np

import starpack


class ContactTests(unittest.TestCase):
  def setUp(self):
    self.base = starpack.make_base_star(5, 2)

  def test_contact_search_finds_interlocking_centres(self):
    state = np.array([[1.0, 1.0, 0.0], [3.1, 1.0, 0.0]])
    state, original_height = starpack.tight_box(state, 1.0, self.base)
    packed, height, changed = starpack.snap_contacts(state, 1.0, self.base, rounds=2)
    self.assertTrue(changed)
    self.assertLess(height, original_height)
    self.assertLess(np.linalg.norm(packed[0, :2] - packed[1, :2]), 1.5)
    self.assertTrue(starpack.exact_valid(packed, height, 1.0, self.base))

  def test_contact_repair_moves_a_chain(self):
    state = np.array([
      [2.0, 2.0, 0.0],
      [4.1, 2.0, 0.0],
      [2.85, 0.8, 0.0],
      [4.175, 0.35, 0.0],
    ])
    state, height = starpack.tight_box(state, 2.0, self.base)
    radius = starpack.pair_contact_radii(0.0, 0.0, self.base)[32]
    pose = state[0].copy()
    pose[:2] = state[1, :2] + (-radius - 1e-7, 0.0)
    result = starpack.repair_blockers(state, 0, pose, height, 2.0, self.base, pinned=1,
                                      reference_polygons=starpack.build_polygons(state, self.base))
    self.assertIsNotNone(result)
    packed, new_height = result
    self.assertTrue(starpack.exact_valid(packed, new_height, 2.0, self.base))
    for index in (2, 3):
      self.assertGreater(np.linalg.norm(packed[index, :2] - state[index, :2]), 1e-5)

  def test_straighten_near_upright_without_expanding_box(self):
    symmetry = 2.0 * math.pi / 5.0
    state = np.array([[1.0, 1.0, 0.02], [3.1, 1.0, -0.015]])
    state, height = starpack.tight_box(state, 1.0, self.base)
    packed, new_height, count = starpack.straighten_stars(
      state, height, 1.0, self.base, symmetry, math.radians(3.0))
    self.assertEqual(count, 2)
    self.assertLessEqual(new_height, height)
    self.assertTrue(starpack.exact_valid(packed, new_height, 1.0, self.base))
    self.assertTrue(np.array_equal(packed[:, 2], np.zeros(2)))

  def test_straighten_wraps_rotational_symmetry(self):
    symmetry = 2.0 * math.pi / 5.0
    state = np.array([[1.25, 1.25, symmetry - 0.02]])
    height = 2.5
    packed, new_height, count = starpack.straighten_stars(
      state, height, 1.0, self.base, symmetry, math.radians(3.0))
    self.assertEqual(count, 1)
    self.assertEqual(packed[0, 2], symmetry)
    self.assertLessEqual(new_height, height)
    self.assertTrue(starpack.exact_valid(packed, new_height, 1.0, self.base))


if __name__ == "__main__":
  unittest.main()
