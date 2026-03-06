import math
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
  s = s.replace("RM", "").replace("rm", "").replace("Rm", "")
  s = s.replace("$", "").replace(",", "").strip()
  return s


def _detect_currency_prefix(items):
  """Return '$' if the receipt uses dollar signs, else ''."""
  for it in items:
    if "$" in it["text"]:
      return "$"
  return ""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _find_nearby_text(items, anchor_idx, patterns, max_lookahead=12):
  # Also search within the anchor item itself
  for p in patterns:
    m = re.search(p, items[anchor_idx]["text"])
    if m:
      return m.group()
  for j in range(anchor_idx + 1, min(len(items), anchor_idx + 1 + max_lookahead)):
    for p in patterns:
      m = re.search(p, items[j]["text"])
      if m:
        return m.group()
  return None


def _line_items(items, y_mid, tol=8):
  return [it for it in items if abs(it["y_mid"] - y_mid) <= tol]


def _money_on_same_line_right(items, anchor_item, tol=8):
  """Return the rightmost money token on the same line, to the right of anchor."""
  line = _line_items(items, anchor_item["y_mid"], tol=tol)
  right = [it for it in line if it["x_mid"] > anchor_item["x_mid"]]
  right.sort(key=lambda d: d["x_mid"])
  for it in right:
    t = _clean_money_token(it["text"])
    if _is_money(t):
      return t
  return None


def _money_in_anchor_text(anchor_item):
  """Extract a money value embedded directly in the anchor's own text (e.g. 'TOTAL: 15.00')."""
  m = re.search(r"\d+\.\d{2}", anchor_item["text"])
  if m:
    v = m.group()
    if _is_money(v):
      return v
  return None


def _money_in_next_lines(items, anchor_item, max_lines=4, tol=8):
  """Scan lines below anchor_item and return the rightmost money on the first valid line."""
  y0 = anchor_item["y_mid"]
  # Slightly wider window to handle varied line heights
  below = [it for it in items if it["y_mid"] > y0 and it["y_mid"] <= y0 + (max_lines * 30)]
  below.sort(key=lambda d: (d["y_mid"], d["x_mid"]))

  lines_used = 0
  current_y = None
  line = []
  prev_line = []

  def best_money(line_items, prev_items):
    line_items.sort(key=lambda d: d["x_mid"])
    lt = " ".join(x["text"].lower() for x in line_items)
    pt = " ".join(x["text"].lower() for x in prev_items)
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


# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------

def extract_date(items):
  date_patterns = [
    r"\d{2}/\d{2}/\d{4}",          # DD/MM/YYYY
    r"\d{4}-\d{2}-\d{2}",          # YYYY-MM-DD
    r"\d{2}-\d{2}-\d{4}",          # DD-MM-YYYY
    r"\d{1,2}\s+\w+\s+\d{4}",      # 05 MAR 2018
    r"\d{2}/\d{2}/\d{2}",          # DD/MM/YY
    r"\d{2}-\d{2}-\d{2}",          # DD-MM-YY
  ]

  _YY_PATTERNS = {r"\d{2}/\d{2}/\d{2}", r"\d{2}-\d{2}-\d{2}"}

  def _plausible_yy(match_str):
    """Guard for YY patterns: at least one of the first two numbers must be ≤ 12."""
    nums = list(map(int, re.findall(r"\d+", match_str)))
    a, b = nums[0], nums[1]
    return (1 <= a <= 31 and 1 <= b <= 12) or (1 <= b <= 31 and 1 <= a <= 12)

  # 1) keyword anchor
  for i, it in enumerate(items):
    if "date" in it["text"].lower():
      candidate = _find_nearby_text(items, i, date_patterns)
      if candidate:
        return candidate

  # 2) fallback: first plausible date in all text
  for it in items:
    for p in date_patterns:
      m = re.search(p, it["text"])
      if m:
        s = m.group()
        if p in _YY_PATTERNS and not _plausible_yy(s):
          continue
        return s

  return None


def _gst_corrected_total(items, best_val):
  """
  If the receipt shows an explicit GST line with amount G, and best_val + G
  matches another money value on the page, return that value (the true
  GST-inclusive total).  Also falls back to inferring 6% GST when no
  explicit GST line is present.  Returns None if no correction is warranted.
  """
  # Collect all money values on the page larger than best_val (potential true totals)
  page_money = set()
  for it in items:
    mv = _clean_money_token(it["text"])
    if _is_money(mv):
      v = float(mv)
      if v > best_val:
        page_money.add(round(v, 2))

  if not page_money:
    return None

  # Pass 1: explicit GST/tax line on the receipt
  gst_amounts = []
  for it in items:
    lt = it["text"].lower()
    if not ("gst" in lt or "tax" in lt):
      continue
    line = _line_items(items, it["y_mid"])
    for x in line:
      mv = _clean_money_token(x["text"])
      if _is_money(mv):
        g = float(mv)
        if 0.01 < g < best_val * 0.5:
          gst_amounts.append(g)

  for g in gst_amounts:
    target = round(best_val + g, 2)
    for candidate in page_money:
      if abs(candidate - target) < 0.02:
        return candidate

  # Pass 2: infer GST at standard Malaysian rates (6% or 8%)
  # If best_val + rate% ≈ a value that actually exists on the page, use it.
  for rate in (0.06, 0.08):
    inferred_gst = round(best_val * rate, 2)
    target = round(best_val + inferred_gst, 2)
    for candidate in page_money:
      if abs(candidate - target) < 0.03:
        return candidate

  return None


# ---------------------------------------------------------------------------
# Total extraction
# ---------------------------------------------------------------------------

def extract_total(items):
  if not items:
    return None

  def line_text(it):
    line = _line_items(items, it["y_mid"])
    line.sort(key=lambda d: d["x_mid"])
    return " ".join(x["text"].lower() for x in line), line

  # Keywords in priority order (higher score = preferred)
  anchors = [
    ("total sales (inclusive", 150),
    ("total sales inclusive", 150),
    ("inclusive of gst", 148),
    ("total (incl. gst", 148),
    ("total (incl gst", 148),
    ("total incl gst", 148),
    ("total including gst", 148),
    ("total inc gst", 148),
    ("rounded total", 135),
    ("nett total", 130),
    ("net total", 130),
    ("amount due", 125),
    ("grand total", 120),
    ("total (rm", 115),
    ("total rm", 115),
    ("jumlah rm", 115),
    ("jumlah", 112),           # Malay for "total"
    ("total sales", 105),
    ("total", 90),
    ("nett", 85),
    ("net", 80),
  ]

  # Lines that should never be treated as the final total
  hard_bad = [
    "cash", "change",
    "gst", "tax",
    "sub total", "subtotal",
    "rounding", "adjust", "adj",
    "discount", "disc",
    "service charge", "svc",
    "tips", "tip", "qty",
    "quantity", "excluding",
    "exclud",
  ]

  candidates = []

  for it in items:
    txt = it["text"].lower()
    for kw, base_score in anchors:
      if kw in txt:
        is_inclusive_kw = kw in {
          "total sales (inclusive", "total sales inclusive", "inclusive of gst",
          "total (incl. gst", "total (incl gst", "total incl gst",
          "total including gst", "total inc gst",
        }

        lt, line = line_text(it)

        # Skip if line above looks like cash/change
        prev = _line_items(items, it["y_mid"] - 14)
        prev_text = " ".join(x["text"].lower() for x in prev)
        if "cash" in prev_text or "change" in prev_text:
          continue

        # is_inclusive: any wording that means "total including GST"
        _incl_words = ("inclusive", "incl.", "incl ", "including", " inc ")
        is_inclusive = is_inclusive_kw or (
          any(w in lt for w in _incl_words) and "gst" in lt
        )

        # Reject bad lines (unless it's an inclusive GST total)
        if any(b in lt for b in hard_bad) and not is_inclusive:
          continue

        # Try to get the money value: same-line right → embedded in text → next lines
        v = _money_on_same_line_right(items, it)
        if v is None:
          v = _money_in_anchor_text(it)
        if v is None:
          v = _money_in_next_lines(items, it, max_lines=4)

        # Last resort: rightmost money anywhere on that line
        if v is None:
          line_money = []
          for x in line:
            mv = _clean_money_token(x["text"])
            if _is_money(mv):
              line_money.append((x["x_mid"], float(mv)))
          if not line_money:
            continue
          line_money.sort(key=lambda z: z[0])
          v = f"{line_money[-1][1]:.2f}"

        try:
          val = float(v)
        except Exception:
          continue

        # Filter out clearly-wrong values
        if val <= 0:
          continue
        # Only skip tiny values for non-priority keywords
        if val < 2.0 and base_score < 110:
          continue

        y_max = max(x["y_mid"] for x in items) + 1e-6
        score = base_score
        score += (it["y_mid"] / y_max) * 30
        # Stronger value signal: log-scale so 153 beats 8 clearly
        score += math.log1p(val) * 0.5

        candidates.append((score, val))
        break  # matched highest-priority keyword for this item

  if candidates:
    candidates.sort(key=lambda x: x[0])
    best = candidates[-1][1]
    corrected = _gst_corrected_total(items, best)
    val = corrected if corrected is not None else best
    return f"{val:.2f}"

  # ── Fallback: score every money token by position + value ──────────────────
  scored = []
  y_max = max(x["y_mid"] for x in items) + 1e-6

  for it in items:
    lt, line = line_text(it)
    if any(b in lt for b in [
      "cash", "change", "gst", "tax",
      "sub total", "subtotal", "rounding",
      "discount", "qty", "quantity"
    ]):
      continue

    mv = _clean_money_token(it["text"])
    if not _is_money(mv):
      continue

    val = float(mv)
    if val <= 0 or val < 2.0:
      continue

    score = (it["y_mid"] / y_max) * 20 + min(val, 9999.0) * 0.002
    scored.append((score, val))

  if not scored:
    return None

  scored.sort(key=lambda x: x[0])
  best = scored[-1][1]
  corrected = _gst_corrected_total(items, best)
  val = corrected if corrected is not None else best
  return f"{val:.2f}"


# ---------------------------------------------------------------------------
# Vendor extraction
# ---------------------------------------------------------------------------

# Malay/Malaysian business suffixes and type words
_BIZ_KEYWORDS = [
  "sdn", "bhd", "s/b",          # Sendirian Berhad variants
  "enterprise", "enterprises",
  "trading", "restaurant", "restoran",
  "hardware", "mart", "supermarket", "hypermarket",
  "store", "stores", "shop", "shops",
  "company", "co.", "ltd", "limited", "pte",
  "perniagaan",   # Malay: "business / trading"
  "kedai",        # Malay: "shop"
  "perusahaan",   # Malay: "enterprise"
  "industri",
  "pharmacy", "clinic", "medical",
  "hotel", "cafe", "bakery",
  "electrical", "electronics", "electric",
  "gift", "gifts", "deco", "decor",
  "beauty", "fashion", "optical", "vision",
  "food", "kopitiam", "mamak",
  "stationery", "stationary", "books",
]

_BAD_CONTAINS = [
  "date", "time", "cashier", "member", "tel:", "phone",
  "tax", "gst", "qty", "amount", "total", "change", "cash",
  "invoice", "receipt", "bill", "ref no", "no:",
  "thank you", "terima kasih",
  "www.", "http",
]


def _vendor_score(text, rank, n_header):
  """
  Score a candidate vendor line.
  rank  : 0-based position in top-to-bottom order within the header region
  n_header : total items in header region
  """
  t = text.strip()
  tl = t.lower()
  s = 0.0

  # ── Position: strong bonus for being near the very top ──────────────────
  # Rank 0 = first box on receipt.  Bonus decays quickly.
  position_bonus = max(0.0, 20.0 - rank * 3.5)
  s += position_bonus

  # ── Business-type keywords: strong signal ───────────────────────────────
  for k in _BIZ_KEYWORDS:
    if k in tl:
      s += 12
      break   # one match is enough

  # ── Penalise meta-info lines ────────────────────────────────────────────
  for b in _BAD_CONTAINS:
    if b in tl:
      s -= 6

  # ── Length preference (vendor names are typically 10-50 chars) ──────────
  length = len(t)
  if 6 <= length <= 60:
    s += length * 0.12
  elif length > 60:
    s -= (length - 60) * 0.15   # very long lines are usually addresses

  # ── Penalise address-like content ───────────────────────────────────────
  digit_count = sum(ch.isdigit() for ch in t)
  if digit_count >= 5:
    s -= 10
  elif digit_count >= 3:
    s -= 4

  if t.count(",") >= 2:
    s -= 5

  # Postcodes (5 consecutive digits) → definitely an address
  if re.search(r"\b\d{5}\b", t):
    s -= 15

  # Lines that are pure punctuation / very short → not a vendor name
  if length < 3:
    s -= 100

  # Pure-number lines
  if re.fullmatch(r"[\d\s\.\-/]+", t):
    s -= 30

  return s


_REG_STRIP_RE = re.compile(
  # Any trailing (...) that contains 4+ consecutive digits — always a reg number,
  # never a legitimate place-name like (JOHOR) which has no digits.
  r'\s*\([^)]*\d{4,}[^)]*\)\s*$',
  re.IGNORECASE
)

def _strip_registration(text):
  """Remove trailing company-registration parentheticals.
  Matches: (519537-X)  (CO.REG : 933109-X)  (126926-H)
  Does NOT match: (JOHOR)  (TAMAN DAYA)  (M)  — no 4-digit run.
  Applied iteratively in case two reg numbers appear back to back.
  """
  prev = None
  while prev != text:
    prev = text
    text = _REG_STRIP_RE.sub("", text).strip()
  return text


def _merge_continuation(items, base_text, anchor_item, tol=12, max_extra_lines=2):
  """
  Try to stitch continuation OCR lines onto base_text.

  base_text    : the already-merged text of the winning line (may differ from
                 anchor_item["text"] which is just one OCR box)
  anchor_item  : representative item whose y_mid marks the winning line
  max_extra_lines : how many extra lines below to attempt to merge
  """
  connectors = ("&", "-", "/", "AND", "and")
  text = base_text.strip()
  y0 = anchor_item["y_mid"]

  for _ in range(max_extra_lines):
    below = sorted(
      [it for it in items if it["y_mid"] > y0 and it["y_mid"] <= y0 + 55],
      key=lambda d: d["y_mid"]
    )
    if not below:
      break

    next_line_items = _line_items(items, below[0]["y_mid"], tol=tol)
    next_line_items.sort(key=lambda d: d["x_mid"])
    next_text = " ".join(x["text"].strip() for x in next_line_items).strip()

    # Strip registration numbers from the candidate continuation line before
    # using it — prevents (126926-H) from being merged in via open-paren trigger
    next_text_clean = _strip_registration(next_text).strip()
    # If stripping left nothing meaningful, stop merging
    if not next_text_clean or next_text_clean in (".", "-", "/"):
      break
    next_text = next_text_clean
    next_lower = next_text.lower()

    is_connector_end = any(text.rstrip().endswith(c) for c in connectors)
    is_biz_continuation = any(k in next_lower for k in _BIZ_KEYWORDS)
    # Merge if current text has an unclosed parenthesis (e.g. "(SETIA")
    is_open_paren = text.count("(") > text.count(")")

    bad_next = any(b in next_lower for b in _BAD_CONTAINS)
    has_postcode = bool(re.search(r"\b\d{5}\b", next_text))

    if (is_connector_end or is_biz_continuation or is_open_paren) and not bad_next and not has_postcode:
      text = (text + " " + next_text).strip()
      y0 = below[0]["y_mid"]   # advance for next iteration
    else:
      break

  return text


def extract_vendor(items):
  if not items:
    return None

  y_max_all = max(it["y_max"] for it in items)
  # Scan top 30% of receipt for the vendor name (tightened from 35%)
  header_cut = 0.30 * y_max_all
  header = [it for it in items if it["y_mid"] <= header_cut]
  header.sort(key=lambda d: (d["y_mid"], d["x_mid"]))

  # Deduplicate by y-line so we score lines, not individual tokens
  def group_into_lines(box_list, tol=8):
    lines = []
    used = set()
    for i, it in enumerate(box_list):
      if i in used:
        continue
      group = [j for j, jt in enumerate(box_list)
               if abs(jt["y_mid"] - it["y_mid"]) <= tol]
      for g in group:
        used.add(g)
      line_items = [box_list[g] for g in group]
      line_items.sort(key=lambda d: d["x_mid"])
      merged_text = " ".join(x["text"].strip() for x in line_items).strip()
      # Use leftmost item as the "representative" for position tracking
      lines.append((merged_text, line_items[0]))
    return lines

  lines = group_into_lines(header)

  candidates = []
  for rank, (text, rep_item) in enumerate(lines):
    t = text.strip()
    if not t:
      continue
    # Skip lines that are pure parenthetical annotations
    if t.startswith("(") and t.endswith(")"):
      continue

    # Score on the cleaned (reg-stripped) text so digit-count penalty doesn't
    # hurt lines like "AEON CO. (M) BHD (126926-H)"
    t_for_score = _strip_registration(t)
    sc = _vendor_score(t_for_score, rank, len(lines))
    candidates.append((sc, t, rep_item))

  if not candidates:
    # Absolute fallback: first non-empty line anywhere
    for it in items:
      t = it["text"].strip()
      if t:
        return t
    return None

  candidates.sort(key=lambda x: x[0], reverse=True)
  best_score, best_text, best_item = candidates[0]

  # Stitch any continuation lines (passes the already-merged line text)
  best_text = _merge_continuation(items, best_text, best_item)

  # Remove trailing company-registration parentheticals
  best_text = _strip_registration(best_text)

  return best_text