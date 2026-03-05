import re
from pathlib import Path


def load_boxes_with_pos(path):
  lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
  items = []

  for line in lines:
    parts = line.split(",")
    coords = list(map(int, parts[:8]))
    text = ",".join(parts[8:]).strip()

    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    items.append({
      "text": text,
      "x_min": x_min,
      "x_max": x_max,
      "y_min": y_min,
      "y_max": y_max,
      "y_mid": (y_min + y_max) / 2.0,
      "x_mid": (x_min + x_max) / 2.0
    })

  # top-to-bottom, left-to-right reading order
  items.sort(key=lambda d: (d["y_mid"], d["x_mid"]))
  return items


def _is_money(s):
  return re.fullmatch(r"\d+\.\d{2}", s) is not None


def _clean_money_token(s):
  s = s.replace("RM", "").replace("rm", "")
  s = s.replace(",", "").strip()
  return s

def _nearby_money_candidates(items, anchor_idx, max_lookahead=20, line_tol=8):
  cand = []

  # 1) same line candidates (by y proximity)
  y0 = items[anchor_idx]["y_mid"]
  same_line = [it for it in items if abs(it["y_mid"] - y0) <= line_tol]
  same_line.sort(key=lambda d: d["x_mid"])
  for it in same_line:
    t = _clean_money_token(it["text"])
    if _is_money(t):
      cand.append(float(t))

  # 2) lookahead candidates (reading order)
  for j in range(anchor_idx + 1, min(len(items), anchor_idx + 1 + max_lookahead)):
    t = _clean_money_token(items[j]["text"])
    if _is_money(t):
      cand.append(float(t))

  return cand

def extract_date(items):
  date_patterns = [
    r"\d{2}/\d{2}/\d{4}",
    r"\d{4}-\d{2}-\d{2}",
    r"\d{2}-\d{2}-\d{4}",
    r"\d{2}-\d{2}-\d{2}"
  ]

  # 1) keyword anchor
  for i, it in enumerate(items):
    if "date" in it["text"].lower():
      candidate = _find_nearby_text(items, i, date_patterns)
      if candidate:
        return candidate

  # 2) fallback: first date-like in all text
  for it in items:
    for p in date_patterns:
      m = re.search(p, it["text"])
      if m:
        return m.group()

  return None


# def _money_in_next_lines(items, anchor_item, max_lines=3, tol=8):
#   # collect lines below the anchor (slightly larger y)
#   y0 = anchor_item["y_mid"]

#   # candidate items that are below anchor within a window
#   below = [it for it in items if it["y_mid"] > y0 and it["y_mid"] <= y0 + (max_lines * 25)]
#   below.sort(key=lambda d: (d["y_mid"], d["x_mid"]))

#   # group by line and get rightmost money
#   used_lines = 0
#   current_y = None
#   line = []

#   def best_money(line_items):
#     line_items.sort(key=lambda d: d["x_mid"])
#     lt = " ".join(x["text"].lower() for x in line_items)

#     # skip tender/change lines
#     if "cash" in lt or "change" in lt:
#       return None

#     for it in reversed(line_items):
#       mv = _clean_money_token(it["text"])
#       if _is_money(mv):
#         return mv

#     return None

#   for it in below:
#     if current_y is None:
#       current_y = it["y_mid"]
#       line = [it]
#       continue

#     if abs(it["y_mid"] - current_y) <= tol:
#       line.append(it)
#     else:
#       used_lines += 1
#       v = best_money(line)
#       if v is not None:
#         return v
#       if used_lines >= max_lines:
#         return None
#       current_y = it["y_mid"]
#       line = [it]

#   # last line
#   if line:
#     return best_money(line)

#   return None


def _find_nearby_text(items, anchor_idx, patterns, max_lookahead=12):
  for j in range(anchor_idx + 1, min(len(items), anchor_idx + 1 + max_lookahead)):
    for p in patterns:
      m = re.search(p, items[j]["text"])
      if m:
        return m.group()
  return None


def _line_items(items, y_mid, tol=8):
  return [it for it in items if abs(it["y_mid"] - y_mid) <= tol]


def _money_on_same_line_right(items, anchor_item, tol=8):
  # money tokens on same line, located to the right of the anchor
  line = _line_items(items, anchor_item["y_mid"], tol=tol)
  right = [it for it in line if it["x_mid"] > anchor_item["x_mid"]]
  right.sort(key=lambda d: d["x_mid"])

  for it in right:
    t = _clean_money_token(it["text"])
    if _is_money(t):
      return t

  return None


def _money_in_next_lines(items, anchor_item, max_lines=3, tol=8):
  y0 = anchor_item["y_mid"]

  # items below anchor, within a vertical window
  below = [it for it in items if it["y_mid"] > y0 and it["y_mid"] <= y0 + 120]
  below.sort(key=lambda d: (d["y_mid"], d["x_mid"]))

  lines_used = 0
  current_y = None
  line = []
  prev_line = []
  
  def best_money(line_items, prev_items):
    line_items.sort(key=lambda d: d["x_mid"])

    lt = " ".join(x["text"].lower() for x in line_items)
    pt = " ".join(x["text"].lower() for x in prev_items)

    # if CASH/CHANGE appears on this line OR the previous line, ignore this line's money
    if "cash" in lt or "change" in lt or "cash" in pt or "change" in pt:
      return None

    for it in reversed(line_items):
      mv = _clean_money_token(it["text"])
      if _is_money(mv):
        return mv

    return None

  for it in below:
    if current_y is None:
      current_y = it["y_mid"]
      line = [it]
      continue

    if abs(it["y_mid"] - current_y) <= tol:
      line.append(it)
    else:
      lines_used += 1
      v = best_money(line, prev_line)
      if v is not None:
        return v
      if lines_used >= max_lines:
        return None

      prev_line = line
      current_y = it["y_mid"]
      line = [it]

  if line:
    return best_money(line, prev_line)
  
  return None


def extract_total(items):
  if not items:
    return None

  def line_text(it):
    line = _line_items(items, it["y_mid"])
    line.sort(key=lambda d: d["x_mid"])
    return " ".join(x["text"].lower() for x in line), line

  # keywords in priority order
  anchors = [
    ("total sales (inclusive", 140),
    ("total sales inclusive", 140),
    ("inclusive of gst", 135),
    ("rounded total", 130),
    ("amount due", 120),
    ("grand total", 115),
    ("total (rm", 110),
    ("total rm", 110),
    ("total sales", 100),
    ("total", 90)
  ]

  # lines we should NOT use as final total lines
  hard_bad = [
    "cash", "change",
    "gst", "tax",
    "sub total", "subtotal",
    "rounding", "adjust", "adj",
    "discount", "disc",
    "service", "svc",
    "tips", "tip", "qty",
    "quantity", "excluding", 
    "exclud"
  ]

  candidates = []

  for it in items:
    txt = it["text"].lower()
    for kw, base_score in anchors:
      if kw in txt:
        
        is_inclusive_kw = (kw in ["total sales (inclusive", "total sales inclusive", "inclusive of gst"])
        
        lt, line = line_text(it)

        prev = _line_items(items, it["y_mid"] - 12)
        prev_text = " ".join(x["text"].lower() for x in prev)

        if "cash" in prev_text or "change" in prev_text:
          continue

        is_inclusive = is_inclusive_kw or ("inclusive" in lt and "gst" in lt)

        # reject bad lines
        if any(b in lt for b in hard_bad) and not is_inclusive:
          continue

        # get money from same line to the right of the anchor if possible
        v = _money_on_same_line_right(items, it)
        if v is None:
          v = _money_in_next_lines(items, it, max_lines=3)

        if v is None:
          # fallback: any money on that line, pick rightmost
          line_money = []
          for x in line:
            mv = _clean_money_token(x["text"])
            if _is_money(mv):
              line_money.append((x["x_mid"], float(mv)))
          if not line_money:
            continue
          line_money.sort(key=lambda z: z[0])
          v = f"{line_money[-1][1]:.2f}"

        # final scoring: prefer stronger keywords + later on page + larger value
        try:
          val = float(v)
          if val <= 0:
            continue
        except Exception:
          continue

        # ignore suspiciously small totals (often unit prices)
        if val > 0.0 and val < 2.0:
          continue

        if val <= 0: 
          continue

        score = base_score
        score += (it["y_mid"] / (max(x["y_mid"] for x in items) + 1e-6)) * 15
        score += min(val, 9999.0) * 0.001

        candidates.append((score, val))
        break

  if candidates:
    candidates.sort(key=lambda x: x[0])
    best = candidates[-1][1]
    return f"{best:.2f}"

  # last fallback: max money excluding cash/change/gst/subtotal lines
  scored = []
  y_max = max(x["y_mid"] for x in items) + 1e-6

  for it in items:
    lt, line = line_text(it)

    # exclude clearly wrong contexts
    if any(b in lt for b in ["cash", "change", "gst", "tax", "sub total", "subtotal", "rounding", "discount", "qty", "quantity"]):
      continue

    mv = _clean_money_token(it["text"])
    if not _is_money(mv):
      continue

    val = float(mv)

    # ignore tiny values (unit prices / adjustments)
    if val > 0 and val < 2.0:
      continue

    # score: later on page + larger value
    score = (it["y_mid"] / y_max) * 10 + min(val, 9999.0) * 0.002
    scored.append((score, val))

  if not scored:
    return None

  scored.sort(key=lambda x: x[0])
  best = scored[-1][1]
  return f"{best:.2f}"


def extract_vendor(items):
  if not items:
    return None

  y_max_all = max(it["y_max"] for it in items)
  header_cut = 0.35 * y_max_all  # was too strict before
  header = [it for it in items if it["y_mid"] <= header_cut]
  header.sort(key=lambda d: (d["y_mid"], d["x_mid"]))

  bad_contains = [
    "date", "time", "cashier", "member", "tel", "phone",
    "tax", "gst", "qty", "amount", "total", "change", "cash",
    "invoice", "receipt", "bill", "ref", "no."
  ]

  biz_keywords = [
    "sdn", "bhd", "enterprise", "trading", "restaurant", "restoran",
    "hardware", "mart", "supermarket", "store", "shop", "company", "co",
    "ltd", "limited", "pte"
  ]

  def score_text(t):
    tl = t.lower()
    s = 0.0

    # strong positives
    for k in biz_keywords:
      if k in tl:
        s += 8

    # mild penalty if line looks like meta info
    for b in bad_contains:
      if b in tl:
        s -= 4

    # prefer earlier lines (top of receipt)
    s += 3.0

    # prefer longer (but not crazy long)
    s += min(len(t), 40) * 0.15

    # penalize address-like lines (many digits or commas)
    digit_count = sum(ch.isdigit() for ch in t)
    if digit_count >= 4:
      s -= 6
    if t.count(",") >= 2:
      s -= 4

    # ignore empty
    if len(t.strip()) < 3:
      s -= 100

    return s

  candidates = []
  for it in header:
    t = it["text"].strip()
    if not t:
      continue
    # remove pure parentheses lines
    if t.startswith("(") and t.endswith(")"):
      continue
    candidates.append((score_text(t), t))

  if not candidates:
    # fallback: first non-empty line anywhere
    for it in items:
      t = it["text"].strip()
      if t:
        return t
    return None

  candidates.sort(key=lambda x: x[0])
  return candidates[-1][1]