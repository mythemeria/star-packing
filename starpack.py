import argparse
import json
import math
import os
import re
import multiprocessing as mp

import numpy as np
from numba import njit
from PIL import Image, ImageDraw


DEFAULT_ASPECT = 1.0
US_FLAG_ASPECT = 1.9
US_CANTON_ASPECT = 0.76 * 13.0 / 7.0
US_FLAG_RED = "#B22234"
US_FLAG_BLUE = "#3C3B6E"
US_FLAG_WHITE = "#FFFFFF"
OVERLAP_WEIGHT = 20.0
CLEARANCE = 0.0
CONTACT_RANGE = 0.25
CONTACT_WEIGHT = 0.12
CENTER_RANGE = 2.25
CENTER_WEIGHT = 0.12


class SerialComm:
  def Get_rank(self):
    return 0

  def Get_size(self):
    return 1

  def bcast(self, value, root=0):
    return value

  def allgather(self, value):
    return [value]

  def Bcast(self, value, root=0):
    return None

  def Barrier(self):
    return None


def communicator():
  try:
    from mpi4py import MPI
    return MPI.COMM_WORLD
  except (ImportError, RuntimeError):
    if any(int(os.environ.get(key, "1")) > 1 for key in (
      "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MV2_COMM_WORLD_SIZE", "SLURM_NTASKS")):
      raise RuntimeError("An MPI launcher is active but mpi4py cannot load its MPI library")
    return SerialComm()


def available_workers():
  count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
  try:
    with open("/sys/fs/cgroup/cpu.max") as stream:
      quota, period = stream.read().split()[:2]
    if quota != "max":
      count = min(count, max(1, int(quota) // int(period)))
  except (OSError, ValueError):
    pass
  if os.environ.get("SLURM_CPUS_PER_TASK"):
    count = min(count, max(1, int(os.environ["SLURM_CPUS_PER_TASK"])))
  return max(1, count)


def parse_symbol(value):
  value = value.strip()
  if value.startswith("{") or value.endswith("}"):
    if not (value.startswith("{") and value.endswith("}")):
      raise argparse.ArgumentTypeError("Use a SchlÃƒÂ¤fli symbol such as {5/2} or 5/2")
    value = value[1:-1]

  match = re.fullmatch(r"([0-9]+)\s*/\s*([0-9]+)", value)
  if match is None:
    raise argparse.ArgumentTypeError("Use a SchlÃƒÂ¤fli symbol such as {5/2} or 5/2")

  n, k = map(int, match.groups())
  if n < 5 or k < 2 or 2 * k >= n:
    raise argparse.ArgumentTypeError("A star requires n >= 5 and 2 <= k < n/2")

  return n, k


def parse_hex_color(value):
  code = value.removeprefix("#")
  if re.fullmatch(r"[0-9a-fA-F]{3}", code):
    code = "".join(channel * 2 for channel in code)
  if not re.fullmatch(r"[0-9a-fA-F]{6}", code):
    raise argparse.ArgumentTypeError("Expected a hex colour such as #00205B or #FFF")
  return f"#{code.upper()}"


def make_base_star(n, k):
  # Outline of the filled star: consecutive outer tips and chord intersections.
  inner_radius = math.cos(math.pi * k / n) / math.cos(math.pi * (k - 1) / n)
  points = np.empty((2 * n, 2), dtype=np.float64)
  for j in range(2 * n):
    angle = math.pi / 2.0 + j * math.pi / n
    radius = 1.0 if j % 2 == 0 else inner_radius
    points[j] = (radius * math.cos(angle), radius * math.sin(angle))

  return points


@njit(cache=True)
def orient(ax, ay, bx, by, cx, cy):
  return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


@njit(cache=True)
def proper_intersection(a, b, c, d):
  o1 = orient(a[0], a[1], b[0], b[1], c[0], c[1])
  o2 = orient(a[0], a[1], b[0], b[1], d[0], d[1])
  o3 = orient(c[0], c[1], d[0], d[1], a[0], a[1])
  o4 = orient(c[0], c[1], d[0], d[1], b[0], b[1])
  return ((o1 > 0.0 and o2 < 0.0) or (o1 < 0.0 and o2 > 0.0)) and ((o3 > 0.0 and o4 < 0.0) or (o3 < 0.0 and o4 > 0.0))


@njit(cache=True)
def point_in_polygon(px, py, poly):
  inside = False
  k = len(poly) - 1

  for j in range(len(poly)):
    x1, y1 = poly[j, 0], poly[j, 1]
    x2, y2 = poly[k, 0], poly[k, 1]
    if (y1 > py) != (y2 > py):
      if px < (x2 - x1) * (py - y1) / (y2 - y1) + x1:
        inside = not inside
    k = j

  return inside


@njit(cache=True)
def polygons_overlap(a, b):
  aminx, amaxx = a[0, 0], a[0, 0]
  aminy, amaxy = a[0, 1], a[0, 1]
  bminx, bmaxx = b[0, 0], b[0, 0]
  bminy, bmaxy = b[0, 1], b[0, 1]

  for j in range(1, len(a)):
    aminx, amaxx = min(aminx, a[j, 0]), max(amaxx, a[j, 0])
    aminy, amaxy = min(aminy, a[j, 1]), max(amaxy, a[j, 1])
    bminx, bmaxx = min(bminx, b[j, 0]), max(bmaxx, b[j, 0])
    bminy, bmaxy = min(bminy, b[j, 1]), max(bmaxy, b[j, 1])

  if amaxx <= bminx or bmaxx <= aminx or amaxy <= bminy or bmaxy <= aminy:
    return False

  for j in range(len(a)):
    for k in range(len(b)):
      if proper_intersection(a[j], a[(j + 1) % len(a)], b[k], b[(k + 1) % len(b)]):
        return True

  if point_in_polygon(a[0, 0], a[0, 1], b) or point_in_polygon(b[0, 0], b[0, 1], a):
    return True

  for j in range(len(a)):
    x = (a[j, 0] + a[(j + 1) % len(a), 0]) * 0.5
    y = (a[j, 1] + a[(j + 1) % len(a), 1]) * 0.5
    if point_in_polygon(x, y, b):
      return True
  for j in range(len(b)):
    x = (b[j, 0] + b[(j + 1) % len(b), 0]) * 0.5
    y = (b[j, 1] + b[(j + 1) % len(b), 1]) * 0.5
    if point_in_polygon(x, y, a):
      return True
  return False


@njit(cache=True)
def point_segment_distance_sq(px, py, ax, ay, bx, by):
  vx, vy = bx - ax, by - ay
  length_sq = vx * vx + vy * vy
  t = 0.0 if length_sq == 0.0 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / length_sq))
  dx, dy = px - ax - t * vx, py - ay - t * vy
  return dx * dx + dy * dy


@njit(cache=True)
def polygon_distance_sq(a, b):
  best = 1e100

  for j in range(len(a)):
    for k in range(len(b)):
      m = (k + 1) % len(b)
      best = min(best, point_segment_distance_sq(a[j, 0], a[j, 1], b[k, 0], b[k, 1], b[m, 0], b[m, 1]))
      m = (j + 1) % len(a)
      best = min(best, point_segment_distance_sq(b[k, 0], b[k, 1], a[j, 0], a[j, 1], a[m, 0], a[m, 1]))

  return best


@njit(cache=True)
def make_polygon(pose, base):
  poly = np.empty((len(base), 2), dtype=np.float64)
  cosine, sine = math.cos(pose[2]), math.sin(pose[2])

  for j in range(len(base)):
    poly[j, 0] = pose[0] + cosine * base[j, 0] - sine * base[j, 1]
    poly[j, 1] = pose[1] + sine * base[j, 0] + cosine * base[j, 1]

  return poly


@njit(cache=True)
def build_polygons(state, base):
  polys = np.empty((len(state), len(base), 2), dtype=np.float64)

  for i in range(len(state)):
    polys[i] = make_polygon(state[i], base)

  return polys


def tight_box(state, aspect, base):
  """Return a translated packing in its smallest box at the fixed aspect."""
  polygons = build_polygons(state, base)
  minimum = polygons.min(axis=(0, 1))
  spans = polygons.max(axis=(0, 1)) - minimum
  height = max(spans[1], spans[0] / aspect) * (1.0 + 1e-10)
  packed = state.copy()
  packed[:, :2] += np.array(((height * aspect - spans[0]) / 2.0,
                             (height - spans[1]) / 2.0)) - minimum
  return packed, height


@njit(cache=True)
def exact_valid(state, height, aspect, base):
  polys = build_polygons(state, base)
  width = height * aspect

  for i in range(len(state)):
    for j in range(len(base)):
      x, y = polys[i, j, 0], polys[i, j, 1]
      if not (0.0 <= x <= width and 0.0 <= y <= height):
        return False
    for k in range(i):
      dx, dy = state[i, 0] - state[k, 0], state[i, 1] - state[k, 1]
      if dx * dx + dy * dy < 4.0 and polygons_overlap(polys[i], polys[k]):
        return False

  return True


@njit(cache=True)
def contact_translations(a, b, max_gap):
  """Nearby vertex-edge contacts in distinct directions, shortest first."""
  moves = np.zeros((4, 2), dtype=np.float64)
  gaps = np.full(4, 1e100, dtype=np.float64)

  for reverse in range(2):
    if reverse == 0:
      source, target, sign = a, b, 1.0
    else:
      source, target, sign = b, a, -1.0
    for j in range(len(source)):
      px, py = source[j, 0], source[j, 1]
      for k in range(len(target)):
        ax, ay = target[k, 0], target[k, 1]
        bx, by = target[(k + 1) % len(target), 0], target[(k + 1) % len(target), 1]
        vx, vy = bx - ax, by - ay
        length_sq = vx * vx + vy * vy
        t = 0.0 if length_sq == 0.0 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / length_sq))
        dx, dy = sign * (ax + t * vx - px), sign * (ay + t * vy - py)
        gap = math.sqrt(dx * dx + dy * dy)
        if not 1e-7 < gap < max_gap:
          continue

        # Keep alternatives that lead to different contacts, rather than four
        # nearly identical vertices on the same side of the other star.
        slot = -1
        for q in range(4):
          if gaps[q] < 1e50 and (dx * moves[q, 0] + dy * moves[q, 1]) > 0.95 * gap * gaps[q]:
            slot = q
            break
        if slot >= 0 and gap >= gaps[slot]:
          continue
        if slot < 0:
          slot = 3
          if gap >= gaps[slot]:
            continue
        for q in range(slot, 3):
          gaps[q], moves[q] = gaps[q + 1], moves[q + 1]
        while slot > 0 and gap < gaps[slot - 1]:
          gaps[slot], moves[slot] = gaps[slot - 1], moves[slot - 1]
          slot -= 1
        gaps[slot], moves[slot, 0], moves[slot, 1] = gap, dx, dy

  return moves, gaps


def contact_count(state, height, aspect, base, tolerance=1e-6):
  polygons = build_polygons(state, base)
  count = 0
  for i, poly in enumerate(polygons):
    minimum, maximum = poly.min(axis=0), poly.max(axis=0)
    count += sum(gap <= tolerance for gap in (minimum[0], minimum[1],
                  height * aspect - maximum[0], height - maximum[1]))
    for j in range(i):
      if np.sum((state[i, :2] - state[j, :2]) ** 2) <= (2.0 + tolerance) ** 2:
        count += 2 * (polygon_distance_sq(poly, polygons[j]) <= tolerance ** 2)
  return count


@njit(cache=True)
def compactness(state):
  """Reward close centres, whether the corresponding contact is a tip or an edge."""
  score = 0.0
  for i in range(len(state)):
    for j in range(i):
      dx, dy = state[i, 0] - state[j, 0], state[i, 1] - state[j, 1]
      gap = CENTER_RANGE - math.sqrt(dx * dx + dy * dy)
      score += max(0.0, gap) ** 2
  return score


@njit(cache=True)
def pair_contact_radii(theta_a, theta_b, base):
  """First nonoverlapping centre radius at each sampled bearing."""
  pose_a = np.array((0.0, 0.0, theta_a))
  pose_b = np.array((0.0, 0.0, theta_b))
  other = make_polygon(pose_b, base)
  radii = np.full(64, 2.000001)
  for bearing in range(64):
    angle = 2.0 * math.pi * bearing / 64.0
    ux, uy = math.cos(angle), math.sin(angle)
    lower = 0.0
    for step in range(1, 33):
      upper = 2.000001 * step / 32.0
      pose_a[0], pose_a[1] = ux * upper, uy * upper
      if not polygons_overlap(make_polygon(pose_a, base), other):
        for _ in range(22):
          middle = (lower + upper) * 0.5
          pose_a[0], pose_a[1] = ux * middle, uy * middle
          if polygons_overlap(make_polygon(pose_a, base), other):
            lower = middle
          else:
            upper = middle
        radii[bearing] = upper
        break
      lower = upper
  return radii


@njit(cache=True)
def cleared_by_translation(poly, polygons, fixed, skip, dx, dy, width, height):
  shifted = poly.copy()
  shifted[:, 0] += dx
  shifted[:, 1] += dy
  for point in shifted:
    if not (0.0 <= point[0] <= width and 0.0 <= point[1] <= height):
      return False
  for j in range(len(polygons)):
    if fixed[j] and j != skip and polygons_overlap(shifted, polygons[j]):
      return False
  return True


@njit(cache=True)
def smallest_blocker_move(poly, polygons, fixed, skip, width, height, max_shift):
  """Find a short sideways or axial displacement clear of the anchored stars."""
  best_score = 1e100
  best_x, best_y = 0.0, 0.0
  for direction in range(24):
    angle = direction * 2.0 * math.pi / 24.0
    ux, uy = math.cos(angle), math.sin(angle)
    limit = max_shift
    for point in poly:
      if ux > 1e-12:
        limit = min(limit, (width - point[0]) / ux)
      elif ux < -1e-12:
        limit = min(limit, -point[0] / ux)
      if uy > 1e-12:
        limit = min(limit, (height - point[1]) / uy)
      elif uy < -1e-12:
        limit = min(limit, -point[1] / uy)
    limit -= 1e-9
    if limit <= 1e-8:
      continue
    lower = 0.0
    for step in range(1, 17):
      upper = limit * step / 16.0
      if cleared_by_translation(poly, polygons, fixed, skip, ux * upper, uy * upper, width, height):
        for _ in range(28):
          middle = (lower + upper) * 0.5
          if cleared_by_translation(poly, polygons, fixed, skip, ux * middle, uy * middle, width, height):
            upper = middle
          else:
            lower = middle
        distance = min(limit, upper + 1e-8)
        shifted = poly.copy()
        shifted[:, 0] += ux * distance
        shifted[:, 1] += uy * distance
        collisions = 0
        for j in range(len(polygons)):
          if j != skip and not fixed[j] and polygons_overlap(shifted, polygons[j]):
            collisions += 1
        score = distance + 0.15 * collisions
        if score < best_score:
          best_score = score
          best_x, best_y = ux * distance, uy * distance
        break
      lower = upper
  return best_x, best_y, best_score < 1e50


@njit(cache=True)
def first_overlapping_pair(state, polygons):
  for a in range(len(state)):
    for b in range(a):
      dx, dy = state[a, 0] - state[b, 0], state[a, 1] - state[b, 1]
      if dx * dx + dy * dy < 4.0 and polygons_overlap(polygons[a], polygons[b]):
        return a, b
  return -1, -1


def repair_blockers(state, i, pose, height, aspect, base, pinned=-1, reference_polygons=None):
  """Keep the intended contact fixed and move obstructing stars recursively."""
  trial = state.copy()
  trial[i] = pose
  polygons = build_polygons(trial, base) if reference_polygons is None else reference_polygons.copy()
  if reference_polygons is not None:
    polygons[i] = make_polygon(pose, base)
  fixed = np.zeros(len(state), dtype=np.bool_)
  fixed[i] = True
  if pinned >= 0:
    fixed[pinned] = True
  if not cleared_by_translation(polygons[i], polygons, fixed, i, 0.0, 0.0, height * aspect, height):
    return None

  move_counts = np.zeros(len(state), dtype=np.int32)
  for _ in range(4 * len(state)):
    a, b = first_overlapping_pair(trial, polygons)
    if a < 0:
      packed, candidate_height = tight_box(trial, aspect, base)
      if candidate_height <= height + 1e-11 and exact_valid(packed, candidate_height, aspect, base):
        return packed, candidate_height
      return None

    if fixed[a] and fixed[b]:
      return None
    blocker, parent = (b, a) if fixed[a] else (a, b)
    move_counts[blocker] += 1
    if move_counts[blocker] > 3:
      return None
    anchors = fixed.copy()
    anchors[parent] = True
    dx, dy, found = smallest_blocker_move(polygons[blocker], polygons, anchors, blocker,
                                          height * aspect, height, 4.0 * CONTACT_RANGE)
    if not found:
      return None
    trial[blocker, :2] += (dx, dy)
    polygons[blocker, :, 0] += dx
    polygons[blocker, :, 1] += dy
  return None


@njit(cache=True)
def contact_pose_fits(pose, i, state, polygons, height, aspect, base):
  poly = make_polygon(pose, base)
  for vertex in poly:
    if not (0.0 <= vertex[0] <= height * aspect and 0.0 <= vertex[1] <= height):
      return False
  for j in range(len(state)):
    if i == j:
      continue
    dx, dy = pose[0] - state[j, 0], pose[1] - state[j, 1]
    if dx * dx + dy * dy < 4.0 and polygons_overlap(poly, polygons[j]):
      return False
  return True


def feasible_contact_move(state, polygons, i, move, height, aspect, base, pinned=-1):
  """Advance towards a contact, stopping at the first obstruction or wall."""
  distance = float(np.linalg.norm(move))
  if distance <= 1e-7:
    return None

  def fits(fraction):
    pose = state[i].copy()
    pose[:2] += move * fraction
    return contact_pose_fits(pose, i, state, polygons, height, aspect, base)

  # A small offset avoids treating floating point boundary ambiguity as an
  # overlap. Binary search turns a blocked target into a reachable tangency.
  full = max(0.0, 1.0 - 1e-9 / distance)
  if not fits(full):
    pose = state[i].copy()
    pose[:2] += move * full
    repaired = repair_blockers(state, i, pose, height, aspect, base, pinned, polygons)
    if repaired is not None:
      return repaired
  if fits(full):
    low = full
  else:
    low, high = 0.0, full
    for _ in range(38):
      middle = (low + high) * 0.5
      if fits(middle):
        low = middle
      else:
        high = middle
  if low * distance <= 1e-7:
    return None
  trial = state.copy()
  trial[i, :2] += move * low
  packed, candidate_height = tight_box(trial, aspect, base)
  if candidate_height > height + 1e-11 or not exact_valid(packed, candidate_height, aspect, base):
    return None
  return packed, candidate_height


def edge_contact_poses(state, polygons, i, j, base, max_move):
  """Generate nearby poses with opposed, coincident star edges."""
  period = 2.0 * math.pi / (len(base) // 2)
  candidates = []
  for a in range(len(base)):
    edge_a = polygons[i, (a + 1) % len(base)] - polygons[i, a]
    angle_a = math.atan2(edge_a[1], edge_a[0])
    for b in range(len(base)):
      edge_b = polygons[j, (b + 1) % len(base)] - polygons[j, b]
      angle_b = math.atan2(edge_b[1], edge_b[0])
      turn = (angle_b + math.pi - angle_a + period / 2.0) % period - period / 2.0
      pose = state[i].copy()
      pose[2] = (pose[2] + turn) % period
      rotated = make_polygon(pose, base)
      midpoint_a = (rotated[a] + rotated[(a + 1) % len(base)]) * 0.5
      midpoint_b = (polygons[j, b] + polygons[j, (b + 1) % len(base)]) * 0.5
      normal = np.array((edge_b[1], -edge_b[0])) / np.linalg.norm(edge_b)
      pose[:2] += midpoint_b + normal * 1e-8 - midpoint_a
      if np.linalg.norm(pose[:2] - state[i, :2]) <= max_move and not polygons_overlap(make_polygon(pose, base), polygons[j]):
        distance = np.linalg.norm(pose[:2] - state[j, :2])
        candidates.append((distance, abs(turn), pose))
  candidates.sort(key=lambda candidate: candidate[:2])
  unique = []
  for _, _, pose in candidates:
    if not any(np.linalg.norm(pose - other) < 1e-6 for other in unique):
      unique.append(pose)
    if len(unique) == 4:
      break
  return unique


def snap_contacts(state, aspect, base, max_gap=CONTACT_RANGE, rounds=2):
  """Compact nearby centres and align edges while preserving a valid box."""
  state, height = tight_box(state, aspect, base)
  if not exact_valid(state, height, aspect, base):
    return state, height, False
  score = compactness(state)
  changed = False
  minimum_cache = {}

  def pair_radii(packing, i, j):
    relative = (packing[i, 2] - packing[j, 2]) % (2.0 * math.pi / (len(base) // 2))
    key = round(relative, 9)
    if key not in minimum_cache:
      minimum_cache[key] = pair_contact_radii(key, 0.0, base)
    return minimum_cache[key]

  def near_pair_minimum(packing, i, j):
    return np.linalg.norm(packing[i, :2] - packing[j, :2]) <= min(pair_radii(packing, i, j)) + 0.01

  for _ in range(rounds):
    sweep_changed = False
    polygons = build_polygons(state, base)
    for i in range(len(state)):
      for j in range(len(state)):
        if i == j or np.sum((state[i, :2] - state[j, :2]) ** 2) > (2.0 + max_gap) ** 2:
          continue
        moves, gaps = contact_translations(polygons[i], polygons[j], max_gap)
        for move, gap in zip(moves, gaps):
          if gap >= max_gap:
            break
          if np.linalg.norm(state[i, :2] + move - state[j, :2]) >= np.linalg.norm(state[i, :2] - state[j, :2]) - 1e-8:
            continue
          result = feasible_contact_move(state, polygons, i, move, height, aspect, base, pinned=j)
          if result is None:
            continue
          packed, trial_height = result
          trial_score = compactness(packed)
          closer = np.linalg.norm(packed[i, :2] - packed[j, :2]) < (
            np.linalg.norm(state[i, :2] - state[j, :2]) - 1e-8)
          if trial_height < height - 1e-9 or (closer and trial_score > score + 1e-9 and
                                               near_pair_minimum(packed, i, j)):
            state, height, score = packed, trial_height, trial_score
            polygons = build_polygons(state, base)
            changed = sweep_changed = True
            break

        # A nearby tip contact need not be the closest way to interlock two
        # concave stars. Jump directly to their sampled closest pair poses.
        radii = pair_radii(state, i, j)
        directions = sorted(range(len(radii)), key=lambda q: radii[q])
        tried = 0
        for direction in directions:
          angle = state[j, 2] + 2.0 * math.pi * direction / len(radii)
          pose = state[i].copy()
          pose[:2] = state[j, :2] + (radii[direction] + 1e-7) * np.array((math.cos(angle), math.sin(angle)))
          if np.linalg.norm(pose[:2] - state[i, :2]) > 1.25:
            continue
          if np.linalg.norm(pose[:2] - state[j, :2]) >= np.linalg.norm(state[i, :2] - state[j, :2]) - 1e-8:
            continue
          tried += 1
          result = repair_blockers(state, i, pose, height, aspect, base, pinned=j,
                                   reference_polygons=polygons)
          if result is not None:
            packed, trial_height = result
            trial_score = compactness(packed)
            if trial_height < height - 1e-9 or trial_score > score + 1e-9:
              state, height, score = packed, trial_height, trial_score
              polygons = build_polygons(state, base)
              changed = sweep_changed = True
              break
          if tried == 6:
            break

        # Parallel opposing edges are a measure-zero target for random angle
        # mutations, so propose those orientations explicitly.
        for pose in edge_contact_poses(state, polygons, i, j, base, 1.25):
          if np.linalg.norm(pose[:2] - state[j, :2]) >= np.linalg.norm(state[i, :2] - state[j, :2]) - 1e-8:
            continue
          result = repair_blockers(state, i, pose, height, aspect, base, pinned=j,
                                   reference_polygons=polygons)
          if result is None:
            continue
          packed, trial_height = result
          trial_score = compactness(packed)
          if trial_height < height - 1e-9 or (trial_score > score + 1e-9 and
                                               near_pair_minimum(packed, i, j)):
            state, height, score = packed, trial_height, trial_score
            polygons = build_polygons(state, base)
            changed = sweep_changed = True
            break

      # A star can also be brought exactly onto a wall without changing angle.
      for axis, limit in ((0, height * aspect), (1, height)):
        for wall in (0.0, limit):
          extent = polygons[i, :, axis].min() if wall == 0.0 else polygons[i, :, axis].max()
          offset = wall - extent
          if not 1e-7 < abs(offset) < max_gap:
            continue
          move = np.zeros(2)
          move[axis] = offset
          result = feasible_contact_move(state, polygons, i, move, height, aspect, base)
          if result is None:
            continue
          packed, trial_height = result
          trial_score = compactness(packed)
          if trial_height < height - 1e-9 or trial_score > score + 1e-9:
            state, height, score = packed, trial_height, trial_score
            polygons = build_polygons(state, base)
            changed = sweep_changed = True

    if not sweep_changed:
      break

  return state, height, changed


def straighten_stars(state, height, aspect, base, symmetry, max_angle):
  """Snap nearly upright stars to exact symmetry angles when the box remains valid."""
  if max_angle <= 0.0:
    return state, height, 0
  state = state.copy()
  polygons = build_polygons(state, base)
  straightened = 0
  tolerance = min(max_angle, symmetry * 0.25)

  for i in range(len(state)):
    offset = (state[i, 2] + symmetry * 0.5) % symmetry - symmetry * 0.5
    if not 1e-10 < abs(offset) <= tolerance:
      continue
    pose = state[i].copy()
    pose[2] = round(pose[2] / symmetry) * symmetry
    trial = state.copy()
    trial[i] = pose
    packed, trial_height = tight_box(trial, aspect, base)
    if trial_height <= height + 1e-11 and exact_valid(packed, trial_height, aspect, base):
      result = packed, trial_height
    else:
      rotated = make_polygon(pose, base) - pose[:2]
      lower = -rotated.min(axis=0)
      upper = np.array((height * aspect, height)) - rotated.max(axis=0)
      if np.any(lower > upper):
        continue
      pose[:2] = np.clip(pose[:2], lower, upper)
      result = repair_blockers(state, i, pose, height, aspect, base,
                               reference_polygons=polygons)
    if result is None:
      continue
    packed, trial_height = result
    if trial_height > height:
      if not exact_valid(packed, height, aspect, base):
        continue
      trial_height = height
    state, height = packed, trial_height
    polygons = build_polygons(state, base)
    straightened += 1

  return state, height, straightened


def polish_candidate(state, aspect, base, best_height, snap_rounds):
  state, height = tight_box(state, aspect, base)
  valid = height < best_height + CONTACT_RANGE and exact_valid(state, height, aspect, base)
  if valid and snap_rounds:
    state, height, _ = snap_contacts(state, aspect, base, rounds=snap_rounds)
  return state, height, valid


@njit(cache=True)
def boundary_penalty(poly, width, height):
  score = 0.0
  minx, maxx = poly[0, 0], poly[0, 0]
  miny, maxy = poly[0, 1], poly[0, 1]

  for j in range(len(poly)):
    x, y = poly[j, 0], poly[j, 1]
    score += min(0.0, x) ** 2 + max(0.0, x - width) ** 2
    score += min(0.0, y) ** 2 + max(0.0, y - height) ** 2
    minx, maxx = min(minx, x), max(maxx, x)
    miny, maxy = min(miny, y), max(maxy, y)

  for gap in (minx, width - maxx, miny, height - maxy):
    score -= CONTACT_WEIGHT * max(0.0, CONTACT_RANGE - max(0.0, gap)) ** 2

  return score


@njit(cache=True)
def overlap_depth(a, b):
  score = 0.01

  for source, target in ((a, b), (b, a)):
    for j in range(len(source)):
      x, y = source[j, 0], source[j, 1]
      if point_in_polygon(x, y, target):
        distance_sq = 1e100
        for k in range(len(target)):
          m = (k + 1) % len(target)
          distance_sq = min(distance_sq, point_segment_distance_sq(x, y, target[k, 0], target[k, 1], target[m, 0], target[m, 1]))
        score += distance_sq

  return score


@njit(cache=True)
def pair_penalty(a, b, dx, dy, overlap_weight, clearance):
  distance_sq = dx * dx + dy * dy
  limit = max(2.0 + clearance, CENTER_RANGE)

  if distance_sq >= limit * limit:
    return 0.0

  if polygons_overlap(a, b):
    return overlap_weight * overlap_depth(a, b)

  gap_sq = polygon_distance_sq(a, b)

  if gap_sq < clearance * clearance:
    return (clearance - math.sqrt(gap_sq)) ** 2
  return -CENTER_WEIGHT * max(0.0, CENTER_RANGE - math.sqrt(distance_sq)) ** 2


@njit(cache=True)
def initial_energy(state, polys, height, aspect, overlap_weight, clearance):
  count = len(state)
  boundary = np.empty(count, dtype=np.float64)
  pairs = np.zeros((count, count), dtype=np.float64)
  energy = 0.0
  for i in range(count):
    boundary[i] = boundary_penalty(polys[i], height * aspect, height)
    energy += boundary[i]
    for j in range(i):
      dx, dy = state[i, 0] - state[j, 0], state[i, 1] - state[j, 1]
      pairs[i, j] = pair_penalty(polys[i], polys[j], dx, dy, overlap_weight, clearance)
      pairs[j, i] = pairs[i, j]
      energy += pairs[i, j]
  return energy, boundary, pairs


@njit(cache=True)
def trial_energy(i, pose, poly, state, polys, height, aspect, energy, boundary, pairs, overlap_weight, clearance):
  new_boundary = boundary_penalty(poly, height * aspect, height)
  new_pairs = np.zeros(len(state), dtype=np.float64)
  proposed = energy + new_boundary - boundary[i]

  for j in range(len(state)):
    if i != j:
      dx, dy = pose[0] - state[j, 0], pose[1] - state[j, 1]
      new_pairs[j] = pair_penalty(poly, polys[j], dx, dy, overlap_weight, clearance)
      proposed += new_pairs[j] - pairs[i, j]
  return proposed, new_boundary, new_pairs


def initialize_state(rng, height, aspect, symmetry, count):
  state = np.empty((count, 3), dtype=np.float64)
  columns = max(1, math.ceil(math.sqrt(count * aspect)))
  rows = math.ceil(count / columns)

  for i in range(count):
    col, row = i % columns, i // columns
    state[i] = ((col + 0.5) * height * aspect / columns, (row + 0.5) * height / rows, rng.uniform(0.0, symmetry))

  state[:, :2] += rng.normal(0.0, 0.05, (count, 2))
  return state


def anneal(rng, state, height, aspect, iterations, initial_temp, base, symmetry, contact_steps):
  state = state.copy()
  polys = build_polygons(state, base)
  energy, boundary, pairs = initial_energy(state, polys, height, aspect, OVERLAP_WEIGHT, CLEARANCE)
  best_state, best_energy = state.copy(), energy
  first_valid = 0 if exact_valid(state, height, aspect, base) else None
  best_valid_state = state.copy() if first_valid is not None else None
  best_valid_height = tight_box(state, aspect, base)[1] if first_valid is not None else math.inf
  best_valid_energy = energy

  for step in range(iterations):
    fraction = step / max(1, iterations - 1)
    temperature = initial_temp * math.exp(-8.0 * fraction)
    position_sigma = 0.25 * (1.0 - fraction) + 0.002
    angle_sigma = 0.25 * (1.0 - fraction) + 0.001
    if first_valid is not None and contact_steps:
      contact_fraction = min(1.0, (step + 1 - first_valid) / contact_steps)
      temperature = min(temperature, 0.003 * math.exp(-4.0 * contact_fraction))
      position_sigma = min(position_sigma, 0.07 * (1.0 - contact_fraction) + 0.002)
      angle_sigma = min(angle_sigma, 0.05 * (1.0 - contact_fraction) + 0.001)
    i = int(rng.integers(len(state)))
    pose = state[i].copy()
    pose[:2] += rng.normal(0.0, position_sigma, 2)
    pose[2] = (pose[2] + rng.normal(0.0, angle_sigma)) % symmetry
    pose[0] = np.clip(pose[0], -1.0, aspect * height + 1.0)
    pose[1] = np.clip(pose[1], -1.0, height + 1.0)
    poly = make_polygon(pose, base)
    proposed, new_boundary, new_pairs = trial_energy(i, pose, poly, state, polys, height, aspect, energy, boundary, pairs, OVERLAP_WEIGHT, CLEARANCE)
    delta = proposed - energy

    if delta <= 0.0 or rng.random() < math.exp(-delta / max(temperature, 1e-12)):
      state[i], polys[i], boundary[i] = pose, poly, new_boundary
      pairs[i, :], pairs[:, i] = new_pairs, new_pairs
      energy = proposed

      if energy < best_energy - 1e-12:
        best_state, best_energy = state.copy(), energy

      check_valid = (first_valid is None and (energy < 1e-10 or (step + 1) % 1000 == 0)) or (
        first_valid is not None and (step + 1) % 100 == 0)
      if check_valid:
        if exact_valid(state, height, aspect, base):
          packed, packed_height = tight_box(state, aspect, base)
          if packed_height < best_valid_height - 1e-10 or (
            abs(packed_height - best_valid_height) <= 1e-10 and energy < best_valid_energy):
            best_valid_state, best_valid_height, best_valid_energy = packed, packed_height, energy
          if first_valid is None:
            first_valid = step + 1

    if (step + 1) % 10000 == 0:
      energy, boundary, pairs = initial_energy(state, polys, height, aspect, OVERLAP_WEIGHT, CLEARANCE)
    if first_valid is not None and step + 1 - first_valid >= contact_steps:
      break

  if best_valid_state is not None:
    return best_valid_state, best_valid_energy, True
  valid = exact_valid(best_state, height, aspect, base)
  return best_state, best_energy, valid


def search_worker(task):
  index, total, epoch, seed, state, height, aspect, iterations, base, symmetry, contact_steps, best_height, snap_rounds = task
  local_seed = (seed + 0x9E3779B97F4A7C15 * index + 0xD1B54A32D192ED03 * epoch) & 0xFFFFFFFFFFFFFFFF
  rng = np.random.default_rng(local_seed)
  fraction = index / (total - 1) if total > 1 else 0.5
  temperature = 10.0 ** (-2.0 + 2.0 * fraction)
  candidate, energy, valid = anneal(rng, state, height, aspect, iterations, temperature,
                                    base, symmetry, contact_steps)
  candidate, candidate_height, valid = polish_candidate(candidate, aspect, base, best_height, snap_rounds)
  return candidate, energy, valid, candidate_height


def gpu_jostle(best_state, best_height, aspect, base, steps, rounds, compression, torch, snap_rounds):
  """Resolve contacts by direct position projection, without velocity or rebound."""
  device = torch.device("cuda")
  shape = torch.as_tensor(base, dtype=torch.float64, device=device)
  count = len(best_state)
  eye = torch.eye(count, dtype=torch.bool, device=device)[:, :, None]
  best_state = best_state.copy()
  failed_scale = 1.0

  with torch.no_grad():
    for attempt in range(rounds):
      target_height = best_height * (1.0 - compression * failed_scale)
      width = target_height * aspect
      poses = torch.as_tensor(best_state, dtype=torch.float64, device=device).clone()
      poses[:, :2] *= target_height / best_height

      for _ in range(steps):
        cosine = torch.cos(poses[:, 2])[:, None]
        sine = torch.sin(poses[:, 2])[:, None]
        local_x, local_y = shape[:, 0][None, :], shape[:, 1][None, :]
        rotated = torch.stack((cosine * local_x - sine * local_y,
                               sine * local_x + cosine * local_y), dim=-1)
        polygon = rotated + poses[:, None, :2]
        next_polygon = torch.roll(polygon, shifts=-1, dims=1)
        samples = torch.cat((polygon, (polygon + next_polygon) * 0.5), dim=1)

        # Each sampled boundary point interacts with the closest edge of each
        # other star. A ray crossing test gives the signed contact distance.
        point = samples[:, None, :, None, :]
        edge_a = polygon[None, :, None, :, :]
        edge_b = next_polygon[None, :, None, :, :]
        edge = edge_b - edge_a
        offset = point - edge_a
        fraction = ((offset * edge).sum(-1) / (edge.square().sum(-1) + 1e-18)).clamp(0.0, 1.0)
        nearest = edge_a + fraction[..., None] * edge
        separation = point - nearest
        squared = separation.square().sum(-1)
        distance, index = squared.min(dim=-1)
        distance = torch.sqrt(distance + 1e-24)
        nearest_delta = separation.gather(3, index[..., None, None].expand(-1, -1, -1, 1, 2)).squeeze(3)

        crosses = ((edge_a[..., 1] > point[..., 1]) != (edge_b[..., 1] > point[..., 1])) & (
          point[..., 0] < edge_a[..., 0] + edge[..., 0] *
          (point[..., 1] - edge_a[..., 1]) / (edge[..., 1] + 1e-24)
        )
        inside = (crosses.sum(-1) % 2) == 1
        # Project the deepest sampled penetration for each pair to its closest
        # boundary. Split the relative correction evenly between the stars.
        penetration = distance.masked_fill(~inside | eye, 0.0)
        depth, deepest = penetration.max(dim=2)
        exit_vector = -nearest_delta.gather(2, deepest[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
        indices = torch.arange(count, device=device)
        forward = (depth > depth.T) | ((depth == depth.T) & (indices[:, None] < indices[None, :]))
        pair_shift = torch.where(forward[..., None], exit_vector, -exit_vector.transpose(0, 1))
        pair_shift.masked_fill_((torch.maximum(depth, depth.T) == 0.0)[..., None], 0.0)
        poses[:, :2] += 0.5 * pair_shift.sum(dim=1) * (1.0 + 1e-10)

        # Clamp each star by its actual extremal vertices; this is the shortest
        # wall correction, and no momentum carries it back into the wall.
        lower = -rotated.amin(dim=1)
        upper = torch.as_tensor((width, target_height), dtype=torch.float64, device=device) - rotated.amax(dim=1)
        poses[:, :2] = torch.minimum(torch.maximum(poses[:, :2], lower), upper)

      candidate = poses.cpu().numpy()
      candidate, height, valid = polish_candidate(candidate, aspect, base, best_height, snap_rounds)
      if valid and height < best_height:
        best_state, best_height = candidate, height
        failed_scale = 1.0
        continue
      failed_scale *= 0.5

  return best_state, best_height


def render_packing_image(filename, state, height, aspect, base, image_width=2000,
                         background="#FFFFFF", border="#000000", fill="#808080", us_flag=False):
  image_height = max(1, round(image_width / (US_FLAG_ASPECT if us_flag else aspect)))
  image = Image.new("RGB", (image_width, image_height), US_FLAG_RED if us_flag else background)
  draw = ImageDraw.Draw(image)
  if us_flag:
    for stripe in range(1, 13, 2):
      top = round(stripe * image_height / 13)
      bottom = round((stripe + 1) * image_height / 13)
      draw.rectangle((0, top, image_width - 1, bottom - 1), fill=US_FLAG_WHITE)
    canton_width = round(image_height * 0.76)
    canton_height = round(image_height * 7 / 13)
    canton = Image.new("RGB", (canton_width, canton_height), US_FLAG_BLUE)
    canton_draw = ImageDraw.Draw(canton)
    scale = min(canton_width / (height * aspect), canton_height / height)
    offset_x = (canton_width - height * aspect * scale) / 2
    offset_y = (canton_height - height * scale) / 2
  polys = build_polygons(state, base)

  for poly in polys:
    if us_flag:
      points = [(float(offset_x + x * scale), float(canton_height - offset_y - y * scale)) for x, y in poly]
      canton_draw.polygon(points, fill=US_FLAG_WHITE, outline=US_FLAG_WHITE)
    else:
      points = [(float(x * image_width / (height * aspect)), float(image_height - y * image_height / height)) for x, y in poly]
      draw.polygon(points, fill=fill, outline=border, width=max(1, round(image_width / 1000)))

  if us_flag:
    image.paste(canton, (0, 0))

  temp = filename + ".tmp.png"
  image.save(temp, format="PNG")
  os.replace(temp, filename)


def save_solution(json_filename, image_filename, state, height, aspect, symbol, base, rank, seed,
                  image_width, background, border, fill, us_flag=False):
  obj = {
    "schlafli_symbol": symbol,
    "stars": len(state),
    "circumradius": 1.0,
    "aspect_ratio": aspect,
    "width": aspect * height,
    "height": height,
    "area": aspect * height * height,
    "rank": rank,
    "seed": int(seed),
    "packing": [{"x": float(x), "y": float(y), "theta": float(theta)} for x, y, theta in state],
  }

  temp = json_filename + ".tmp"

  with open(temp, "w") as stream:
    json.dump(obj, stream, indent=2)

  os.replace(temp, json_filename)
  render_packing_image(image_filename, state, height, aspect, base, image_width, background, border, fill, us_flag)


def load_solution(filename, symbol, count):
  with open(filename) as stream:
    obj = json.load(stream)

  if obj.get("stars") != count or len(obj.get("packing", [])) != count:
    raise ValueError(f"Expected {count} stars in {filename}")

  if obj.get("schlafli_symbol") != symbol:
    raise ValueError(f"Loaded SchlÃƒÂ¤fli symbol does not match {symbol}")

  if float(obj.get("circumradius", 1.0)) != 1.0:
    raise ValueError("Loaded circumradius must be 1")

  state = np.array([[float(star[key]) for key in ("x", "y", "theta")] for star in obj["packing"]], dtype=np.float64)
  height = float(obj["height"])
  aspect = float(obj["aspect_ratio"]) if "aspect_ratio" in obj else float(obj["width"]) / height

  if not np.isfinite(state).all() or not math.isfinite(height) or not math.isfinite(aspect) or height <= 0 or aspect <= 0:
    raise ValueError("Loaded packing contains invalid coordinates or dimensions")

  if "width" in obj and not math.isclose(float(obj["width"]), height * aspect, rel_tol=1e-10):
    raise ValueError("Loaded width disagrees with height and aspect_ratio")
  return state, height, aspect


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("symbol", type=parse_symbol, help="SchlÃƒÂ¤fli symbol, e.g. '{5/2}' or 5/2")
  parser.add_argument("count", type=int, help="Number of stars to pack")
  parser.add_argument("--aspect", type=float, default=None, help="Width / height; default 1:1, or loaded aspect")
  parser.add_argument("--us-flag", action="store_true", help="Use the US canton aspect and render white stars on a full US flag; overrides aspect and colours")
  parser.add_argument("--start-height", type=float, default=16.0)
  parser.add_argument("--iterations", type=int, default=100000, help="Annealing iterations per epoch")
  parser.add_argument("--contact-steps", type=int, default=2000, help="Extra annealing steps after the first valid packing")
  parser.add_argument("--snap-rounds", type=int, default=2, help="Nearby contact-snapping sweeps per improvement; 0 disables")
  parser.add_argument("--shrink", type=float, default=0.9995)
  parser.add_argument("--exchange-every", type=int, default=10, help="Print progress every N epochs")
  parser.add_argument("--workers", type=int, default=0, help="Local CPU processes; 0 uses available CPUs when MPI has one rank")
  parser.add_argument("--epochs", type=int, default=0, help="Stop after N epochs; 0 continues indefinitely")
  parser.add_argument("--straighten-every", type=int, default=10, help="Attempt near-upright orientations every N epochs; 0 disables")
  parser.add_argument("--straighten-angle", type=float, default=3.0, help="Maximum angle from upright in degrees")
  parser.add_argument("--jostle-every", type=int, default=10, help="Run GPU physics every N epochs; 0 disables it")
  parser.add_argument("--jostle-steps", type=int, default=200, help="Contact projection steps per compression attempt")
  parser.add_argument("--jostle-rounds", type=int, default=30, help="Compression attempts per GPU pass")
  parser.add_argument("--jostle-compression", type=float, default=0.0003, help="Fractional box shrink per attempt")
  parser.add_argument("--image-width", type=int, default=2000)
  parser.add_argument("--background", type=parse_hex_color, default="#FFFFFF", help="Image background hex colour")
  parser.add_argument("--border", type=parse_hex_color, default="#000000", help="Star outline hex colour")
  parser.add_argument("--fill", type=parse_hex_color, default="#808080", help="Star fill hex colour")
  parser.add_argument("--seed", type=int, default=123456789)
  args = parser.parse_args()

  if args.count <= 0:
    parser.error("count must be positive")
  if args.workers < 0 or args.epochs < 0 or args.straighten_every < 0:
    parser.error("--workers, --epochs and --straighten-every must be nonnegative")
  if not (math.isfinite(args.straighten_angle) and args.straighten_angle >= 0):
    parser.error("--straighten-angle must be finite and nonnegative")

  if args.iterations <= 0 or args.exchange_every <= 0 or args.image_width <= 0:
    parser.error("--iterations, --exchange-every and --image-width must be positive")
  if args.contact_steps < 0 or args.snap_rounds < 0:
    parser.error("--contact-steps and --snap-rounds must be nonnegative")
  if args.jostle_every < 0 or args.jostle_steps <= 0 or args.jostle_rounds <= 0:
    parser.error("--jostle-every must be nonnegative; --jostle-steps and --jostle-rounds must be positive")
  if not (math.isfinite(args.jostle_compression) and 0.0 < args.jostle_compression < 1.0):
    parser.error("--jostle-compression must be finite and between 0 and 1")

  if not 0.0 < args.shrink < 1.0 or args.start_height <= 0:
    parser.error("--shrink must be between 0 and 1, and --start-height must be positive")

  if args.aspect is not None and not (math.isfinite(args.aspect) and args.aspect > 0):
    parser.error("--aspect must be finite and positive")
  if args.us_flag:
    args.aspect = US_CANTON_ASPECT

  n, k = args.symbol
  symbol = f"{{{n}/{k}}}"
  base = make_base_star(n, k)
  symmetry = 2.0 * math.pi / n
  output_dir = "results"
  stem = f"{n}-{k}_{args.count}"
  json_filename = os.path.join(output_dir, f"{stem}{'_us_flag' if args.us_flag else ''}.json")
  image_filename = os.path.join(output_dir, f"{stem}{'_us_flag' if args.us_flag else ''}.png")

  comm = communicator()
  rank, size = comm.Get_rank(), comm.Get_size()
  workers = (args.workers or available_workers()) if size == 1 else 1
  seed = (args.seed + 0x9E3779B97F4A7C15 * rank) & 0xFFFFFFFFFFFFFFFF
  rng = np.random.default_rng(seed)

  torch = None
  if rank == 0 and args.jostle_every:
    try:
      import torch as torch_module
      if torch_module.cuda.is_available():
        torch = torch_module
      else:
        print("GPU jostling disabled: CUDA is unavailable", flush=True)
    except ImportError:
      print("GPU jostling disabled: install PyTorch with CUDA support", flush=True)
  gpu_enabled = comm.bcast(torch is not None if rank == 0 else None, root=0)

  if rank == 0:
    try:
      os.makedirs(output_dir, exist_ok=True)
      loaded_filename = json_filename
      if args.us_flag and not os.path.exists(loaded_filename):
        loaded_filename = os.path.join(output_dir, f"{stem}.json")
      loaded = load_solution(loaded_filename, symbol, args.count) if os.path.exists(loaded_filename) else None
      load_error = None

    except (OSError, ValueError, KeyError, TypeError) as exc:
      loaded, load_error = None, str(exc)

  else:
    loaded, load_error = None, None

  load_error = comm.bcast(load_error, root=0)

  if load_error is not None:
    raise ValueError(load_error)

  loaded = comm.bcast(loaded, root=0)

  if loaded is not None:
    state, height, aspect = loaded
    converted = args.us_flag and not math.isclose(args.aspect, aspect, rel_tol=0, abs_tol=1e-12)
    if converted:
      if not exact_valid(state, height, aspect, base):
        raise ValueError("Loaded packing is not valid")
      aspect = args.aspect
      state, height = tight_box(state, aspect, base)
    elif args.aspect is not None and not math.isclose(args.aspect, aspect, abs_tol=1e-12):
      raise ValueError(f"Loaded aspect {aspect} does not match --aspect {args.aspect}")
    if rank == 0:
      try:
        if not exact_valid(state, height, aspect, base):
          raise ValueError("Loaded packing is not valid")
        state, tight_height = tight_box(state, aspect, base)
        if not exact_valid(state, tight_height, aspect, base):
          raise ValueError("Loaded packing is not valid after tightening")
        tightened = converted or (args.us_flag and not os.path.exists(json_filename)) or tight_height < height
        height = tight_height
        if args.snap_rounds:
          state, height, snapped = snap_contacts(state, aspect, base, rounds=args.snap_rounds)
          tightened = tightened or snapped
        print(f"loaded packing: H={height:.12f} W={height * aspect:.12f} A={height * height * aspect:.12f}", flush=True)
        if tightened:
          save_solution(json_filename, image_filename, state, height, aspect, symbol, base, rank, seed,
                        args.image_width, args.background, args.border, args.fill, args.us_flag)
        else:
          render_packing_image(image_filename, state, height, aspect, base, args.image_width,
                               args.background, args.border, args.fill, args.us_flag)
        load_error = None
      except (OSError, ValueError) as exc:
        load_error = str(exc)

    load_error = comm.bcast(load_error if rank == 0 else None, root=0)
    if load_error is not None:
      raise ValueError(load_error)
    state, height = comm.bcast((state, height) if rank == 0 else None, root=0)
    best_height = height
    best_state = state.copy()
    best_compactness = compactness(state)

    state = state.copy()
    state[:, :2] *= args.shrink
    height *= args.shrink

  else:
    aspect = args.aspect if args.aspect is not None else DEFAULT_ASPECT
    height = args.start_height
    state = initialize_state(rng, height, aspect, symmetry, args.count)
    best_height = math.inf
    best_state = None
    best_compactness = -math.inf

  dummy = initialize_state(rng, max(height, 16.0), aspect, symmetry, args.count)
  dummy_polys = build_polygons(dummy, base)
  initial_energy(dummy, dummy_polys, max(height, 16.0), aspect, OVERLAP_WEIGHT, CLEARANCE)
  trial_energy(0, dummy[0], dummy_polys[0], dummy, dummy_polys, max(height, 16.0), aspect, 0.0, np.zeros(args.count), np.zeros((args.count, args.count)), OVERLAP_WEIGHT, CLEARANCE)
  exact_valid(dummy, max(height, 16.0), aspect, base)
  comm.Barrier()

  pool = mp.get_context("spawn").Pool(workers) if workers > 1 else None
  worker_states = [state.copy() for _ in range(workers)] if pool is not None else None
  if pool is not None and loaded is None:
    for index in range(1, workers):
      worker_states[index] = initialize_state(rng, height, aspect, symmetry, args.count)

  try:
    epoch = 0
    while True:
      if pool is not None:
        tasks = [(index, workers, epoch, args.seed, worker_states[index], height, aspect,
                  args.iterations, base, symmetry, args.contact_steps, best_height, args.snap_rounds)
                 for index in range(workers)]
        results = pool.map(search_worker, tasks)
        selected = min(range(workers), key=lambda index: (
          results[index][3] if results[index][2] else math.inf,
          -compactness(results[index][0]) if results[index][2] else results[index][1]))
        candidate, energy, valid, candidate_height = results[selected]
      else:
        rank_fraction = rank / (size - 1) if size > 1 else 0.5
        temperature = 10.0 ** (-2.0 + 2.0 * rank_fraction)
        candidate, energy, valid = anneal(rng, state, height, aspect, args.iterations, temperature,
                                          base, symmetry, args.contact_steps)
        candidate, candidate_height, valid = polish_candidate(candidate, aspect, base, best_height, args.snap_rounds)
      candidate_compactness = compactness(candidate) if valid else -math.inf
      better = valid and (candidate_height < best_height or (
        candidate_height == best_height and candidate_compactness > best_compactness + 1e-9))
      local_record = (candidate_height if better else math.inf, -candidate_compactness, rank)
      global_height, neg_compactness, global_rank = min(comm.allgather(local_record))

      if math.isfinite(global_height):
        source = candidate.copy() if rank == global_rank else np.empty((args.count, 3), dtype=np.float64)
        comm.Bcast(source, root=global_rank)
        best_height = global_height
        best_state = source.copy()
        best_compactness = -neg_compactness

        if rank == global_rank:
          print(f"new best packing: H={global_height:.12f} W={global_height * aspect:.12f} A={global_height * global_height * aspect:.12f}", flush=True)
          save_solution(json_filename, image_filename, source, global_height, aspect, symbol, base, rank, seed,
                        args.image_width, args.background, args.border, args.fill, args.us_flag)

        height = global_height * args.shrink
        state = source.copy()
        state[:, :2] *= args.shrink

        if rank != global_rank:
          state[:, :2] += rng.normal(0.0, 0.03, (args.count, 2))
          state[:, 2] = (state[:, 2] + rng.normal(0.0, 0.03, args.count)) % symmetry
        if pool is not None:
          for index in range(workers):
            worker_states[index] = state.copy()
            if index != selected:
              worker_states[index][:, :2] += rng.normal(0.0, 0.03, (args.count, 2))
              worker_states[index][:, 2] = (worker_states[index][:, 2] +
                                             rng.normal(0.0, 0.03, args.count)) % symmetry

      else:
        state = candidate
        if pool is not None:
          for index, (worker_candidate, worker_energy, _, _) in enumerate(results):
            worker_states[index] = worker_candidate
            if worker_energy > 1.0 and rng.random() < 0.05:
              worker_states[index] = initialize_state(rng, height, aspect, symmetry, args.count)

        if pool is None and energy > 1.0 and rng.random() < 0.05:
          state = initialize_state(rng, height, aspect, symmetry, args.count)

      epoch += 1

      if gpu_enabled and epoch % args.jostle_every == 0:
        if rank == 0 and best_state is not None:
          jostled, jostled_height = gpu_jostle(best_state, best_height, aspect, base,
                                               args.jostle_steps, args.jostle_rounds, args.jostle_compression,
                                               torch, args.snap_rounds)
          improved = jostled_height < best_height
        else:
          improved = False

        improved = comm.bcast(improved, root=0)
        if improved:
          best_height = comm.bcast(jostled_height if rank == 0 else None, root=0)
          source = jostled.copy() if rank == 0 else np.empty((args.count, 3), dtype=np.float64)
          comm.Bcast(source, root=0)
          best_state = source.copy()
          best_compactness = compactness(source)
          if rank == 0:
            print(f"GPU jostle improved packing: H={best_height:.12f} W={best_height * aspect:.12f}", flush=True)
            save_solution(json_filename, image_filename, source, best_height, aspect, symbol, base, rank, seed,
                          args.image_width, args.background, args.border, args.fill, args.us_flag)
          height = best_height * args.shrink
          state = source.copy()
          state[:, :2] *= args.shrink
          if pool is not None:
            for index in range(workers):
              worker_states[index] = state.copy()
              if index:
                worker_states[index][:, :2] += rng.normal(0.0, 0.03, (args.count, 2))
                worker_states[index][:, 2] = (worker_states[index][:, 2] +
                                               rng.normal(0.0, 0.03, args.count)) % symmetry

      if args.straighten_every and epoch % args.straighten_every == 0:
        if rank == 0 and best_state is not None:
          straightened_state, straightened_height, straightened_count = straighten_stars(
            best_state, best_height, aspect, base, symmetry, math.radians(args.straighten_angle))
        else:
          straightened_count = 0
        changed = comm.bcast(straightened_count > 0, root=0)
        if changed:
          best_height = comm.bcast(straightened_height if rank == 0 else None, root=0)
          source = straightened_state.copy() if rank == 0 else np.empty((args.count, 3), dtype=np.float64)
          comm.Bcast(source, root=0)
          best_state = source.copy()
          best_compactness = compactness(source)
          if rank == 0:
            print(f"straightened {straightened_count} stars: H={best_height:.12f} W={best_height * aspect:.12f}", flush=True)
            save_solution(json_filename, image_filename, source, best_height, aspect, symbol, base, rank, seed,
                          args.image_width, args.background, args.border, args.fill, args.us_flag)
          height = best_height * args.shrink
          state = source.copy()
          state[:, :2] *= args.shrink
          if pool is not None:
            for index in range(workers):
              worker_states[index] = state.copy()
              if index:
                worker_states[index][:, :2] += rng.normal(0.0, 0.03, (args.count, 2))
                worker_states[index][:, 2] = (worker_states[index][:, 2] +
                                               rng.normal(0.0, 0.03, args.count)) % symmetry

      if epoch % args.exchange_every == 0 and rank == 0:
        print(f"epoch={epoch} workers={workers if pool is not None else size} target_H={height:.12f} target_W={height * aspect:.12f} best_H={best_height:.12f}", flush=True)

      if args.epochs and epoch >= args.epochs:
        break
  finally:
    if pool is not None:
      pool.terminate()
      pool.join()


if __name__ == "__main__":
  main()
