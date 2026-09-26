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


if __name__ == "__main__":
  unittest.main()
