import os
import re
import json
import logging
from datetime import datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo
import requests
import pandas as pd
from playwright.sync_api import sync_playwright, Page

logger = logging.getLogger(__name__)

OLLAMA_BASE     = "http://localhost:11434"
PLANNER_MODEL   = "qwen2.5-coder:7b"
EXTRACTOR_MODEL = "qwen2.5:7b"
NAV_MODEL       = "qwen2.5:7b"     
VISION_MODEL    = "llava:13b"       

AI_FIRST = os.environ.get("AI_FIRST", "0") not in ("0", "", "false", "False")
AI_MERGE = os.environ.get("AI_MERGE", "0") not in ("0", "", "false", "False")

AI_PICK_CONFIDENCE = 0.3 if AI_FIRST else 0.4

MST = ZoneInfo("America/Edmonton")


def now_mst() -> datetime:
    return datetime.now(MST)

def mst_stamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    return now_mst().strftime(fmt)

def mst_log_prefix() -> str:
    return now_mst().strftime("%H:%M:%S")




def _safe(text: str, maxlen: int = 28) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", text).strip("_")[:maxlen]

def make_output_dir(site: str, location: str, base: str = "scraper_outputs") -> str:
    path = os.path.join(base, _safe(site), _safe(location))
    os.makedirs(path, exist_ok=True)
    return path


CSV_COLUMNS = [
    "pickup_date", "return_date", "pickup_time", "return_time",
    "car_name", "car_type", "price_per_day", "transmission", "seats", "bags", "location",
]


def _to_iso_date(date_str: str) -> str:
    """Normalize common date strings to YYYY-MM-DD."""
    raw = str(date_str or "").strip()
    if not raw:
        raise ValueError("empty date")
    for fmt in (
        "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%d.%m.%Y",
        "%B %d, %Y", "%b %d, %Y", "%d %b %Y", "%d %B %Y",
    ):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # "Jun 20" without year — assume current/next occurrence in MST
    m = re.match(r"([A-Za-z]{3,9})\s+(\d{1,2})(?:,?\s*(\d{4}))?", raw)
    if m:
        month_s, day_s, year_s = m.group(1), m.group(2), m.group(3)
        year = int(year_s) if year_s else now_mst().year
        try:
            dt = datetime.strptime(f"{month_s} {day_s} {year}", "%b %d %Y")
            if not year_s and dt.date() < now_mst().date():
                dt = dt.replace(year=year + 1)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            try:
                dt = datetime.strptime(f"{month_s} {day_s} {year}", "%B %d %Y")
                if not year_s and dt.date() < now_mst().date():
                    dt = dt.replace(year=year + 1)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
    raise ValueError(f"unrecognized date format: {date_str!r}")


def _normalize_time_str(time_str: str) -> str:
    """Normalize to HH:MM (24h)."""
    raw = str(time_str or "").strip()
    if not raw:
        return ""
    raw = raw.upper().replace(".", ":")
    m = re.match(r"(\d{1,2}):(\d{2})\s*(AM|PM)?", raw, re.I)
    if not m:
        return raw[:5]
    hour, minute, ampm = int(m.group(1)), m.group(2), (m.group(3) or "").upper()
    if ampm == "PM" and hour < 12:
        hour += 12
    elif ampm == "AM" and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute}"


def _split_iso_datetime(val: str) -> tuple[str | None, str | None]:
    raw = str(val or "").strip()
    if not raw:
        return None, None
    if "T" in raw:
        date_part, time_part = raw.split("T", 1)
        return date_part[:10], _normalize_time_str(time_part)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw, None
    try:
        return _to_iso_date(raw), None
    except ValueError:
        return None, None


def _parse_rental_field(val: str) -> tuple[str | None, str | None]:
    """Return (YYYY-MM-DD, HH:MM) from an ISO, US, or display date/time string."""
    raw = str(val or "").strip()
    if not raw:
        return None, None
    if "T" in raw or re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return _split_iso_datetime(raw)
    tm = re.search(r"(\d{1,2}:\d{2}\s*(?:AM|PM)?)", raw, re.I)
    if tm:
        date_part = raw[: tm.start()].strip(" ,|-")
        try:
            return _to_iso_date(date_part), _normalize_time_str(tm.group(1))
        except ValueError:
            return None, _normalize_time_str(tm.group(1))
    try:
        return _to_iso_date(raw), None
    except ValueError:
        return None, _normalize_time_str(raw)


def _merge_rental_metadata(page_meta: dict, user_meta: dict) -> dict:
    """Page preloads win; explicit user CLI values override."""
    out = {k: v for k, v in page_meta.items() if v}
    for key, val in user_meta.items():
        if val:
            out[key] = val
    return out


def _accumulate_rental_metadata(base: dict, extra: dict) -> dict:
    """Merge page reads — later reads fill in missing fields."""
    out = {k: v for k, v in base.items() if v}
    for key, val in extra.items():
        if val:
            out[key] = val
    return out


def _sixt_site_default_dates() -> tuple[str, str]:
    """Sixt.ca preloads pickup ~2 days out, return ~5 days out."""
    now = now_mst()
    return (
        (now + timedelta(days=2)).strftime("%Y-%m-%d"),
        (now + timedelta(days=5)).strftime("%Y-%m-%d"),
    )


def _enterprise_site_default_dates() -> tuple[str, str]:
    """Enterprise preloads pickup tomorrow, return the day after."""
    now = now_mst()
    return (
        (now + timedelta(days=1)).strftime("%Y-%m-%d"),
        (now + timedelta(days=2)).strftime("%Y-%m-%d"),
    )


_SIXT_METADATA_JS = r"""
() => {
  const out = {};
  const readWidgetValue = (tid) => {
    const root = document.querySelector(`[data-testid="${tid}"]`);
    if (!root) return null;
    for (const btn of root.querySelectorAll('button')) {
      const aria = (btn.getAttribute('aria-label') || '').trim();
      const text = (btn.innerText || '').replace(/\s+/g, ' ').trim();
      for (const cand of [aria, text]) {
        if (!cand) continue;
        if (/^(pickup|return)\s+(date|time)$/i.test(cand)) continue;
        return cand;
      }
    }
    const full = (root.innerText || '').replace(/\s+/g, ' ').trim();
    const m = full.match(/(?:pickup|return)\s+(?:date|time)\s*(.+)$/i);
    return m ? m[1].trim() : null;
  };

  const widgets = {
    pickup_date_display: 'rent-search-form-pickup-date-input',
    return_date_display: 'rent-search-form-return-date-input',
    pickup_time_display: 'rent-search-form-pickup-time-input',
    return_time_display: 'rent-search-form-return-time-input',
  };
  for (const [k, tid] of Object.entries(widgets)) {
    const v = readWidgetValue(tid);
    if (v) out[k] = v;
  }

  try {
    const p = JSON.parse(localStorage.getItem('rent-search_historyPickup') || '[]');
    if (p && p[0]) {
      if (p[0].date && !out.pickup_storage) out.pickup_storage = p[0].date;
      if (p[0].time && !out.pickup_time_storage) out.pickup_time_storage = p[0].time;
    }
    const r = JSON.parse(localStorage.getItem('rent-search_historyReturn') || '[]');
    if (r && r[0]) {
      if (r[0].date && !out.return_storage) out.return_storage = r[0].date;
      if (r[0].time && !out.return_time_storage) out.return_time_storage = r[0].time;
    }
  } catch (e) {}

  const html = document.documentElement.innerHTML;
  for (const key of ['pickupDate','returnDate','pickup_date','return_date','pickUpDate']) {
    const re = new RegExp('"' + key + '"\\s*:\\s*"([^"]+)"', 'i');
    const m = html.match(re);
    if (m) out[key] = m[1];
  }

  try {
    const url = new URL(location.href);
    for (const [param, field] of [
      ['pickupDate','url_pickup'], ['returnDate','url_return'],
      ['pu','url_pickup'], ['do','url_return'],
    ]) {
      const v = url.searchParams.get(param);
      if (v) out[field] = decodeURIComponent(v);
    }
  } catch (e) {}

  return out;
}
"""


def _normalize_sixt_metadata(raw: dict) -> dict:
    out: dict[str, str] = {}
    field_map = [
        ("pickup_date_display", "pickup_date", "pickup_time"),
        ("pickupDate", "pickup_date", "pickup_time"),
        ("pickup_date", "pickup_date", "pickup_time"),
        ("url_pickup", "pickup_date", "pickup_time"),
        ("pickup_storage", "pickup_date", "pickup_time"),
        ("return_date_display", "return_date", "return_time"),
        ("returnDate", "return_date", "return_time"),
        ("return_date", "return_date", "return_time"),
        ("url_return", "return_date", "return_time"),
        ("return_storage", "return_date", "return_time"),
        ("pickup_time_display", None, "pickup_time"),
        ("return_time_display", None, "return_time"),
        ("pickup_time_storage", None, "pickup_time"),
        ("return_time_storage", None, "return_time"),
    ]
    for src_key, date_key, time_key in field_map:
        val = raw.get(src_key)
        if not val:
            continue
        iso, tm = _parse_rental_field(val)
        if iso and date_key and date_key not in out:
            out[date_key] = iso
        if tm and time_key and time_key not in out:
            out[time_key] = tm
    return out


def _wait_for_sixt_search_form(page: Page, timeout_ms: int = 12000) -> None:
    """Wait until Sixt's rent-search date widgets hydrate."""
    try:
        page.wait_for_function("""
            () => {
              const read = (tid) => {
                const root = document.querySelector(`[data-testid="${tid}"]`);
                if (!root) return '';
                for (const btn of root.querySelectorAll('button')) {
                  const t = (btn.getAttribute('aria-label') || btn.innerText || '').trim();
                  if (t && !/^(pickup|return)\\s+(date|time)$/i.test(t)) return t;
                }
                return '';
              };
              return read('rent-search-form-pickup-date-input')
                  || read('rent-search-form-return-date-input')
                  || localStorage.getItem('rent-search_historyPickup');
            }
        """, timeout=timeout_ms)
    except Exception:
        pass


def _read_sixt_rental_metadata(page: Page) -> dict:
    try:
        raw = page.evaluate(_SIXT_METADATA_JS)
    except Exception:
        raw = {}
    return _normalize_sixt_metadata(raw or {})


_ENTERPRISE_METADATA_JS = r"""
() => {
  const out = {};
  const readVal = (sels) => {
    for (const sel of sels) {
      const el = document.querySelector(sel);
      if (!el) continue;
      const val = (el.value || el.getAttribute('value') || '').trim();
      if (val) return val;
    }
    return null;
  };
  const readSelect = (sels) => {
    for (const sel of sels) {
      const el = document.querySelector(sel);
      if (!el || el.tagName.toLowerCase() !== 'select') continue;
      const opt = el.options[el.selectedIndex];
      const val = (el.value || (opt && opt.value) || '').trim();
      const label = (opt && opt.text || '').trim();
      if (val) return val;
      if (label) return label;
    }
    return null;
  };
  out.pickup_date_raw = readVal([
    'input#pickupDate', 'input#from-date',
    'input[id*="pickup"][id*="date" i]', 'input[name*="pickup" i][name*="date" i]',
  ]);
  out.return_date_raw = readVal([
    'input#returnDate', 'input#to-date',
    'input[id*="return"][id*="date" i]', 'input[name*="return" i][name*="date" i]',
  ]);
  out.pickup_time_raw = readSelect([
    'select#pickupTime', 'select#from-time',
    'select[id*="pickup"][id*="time" i]', 'select[name*="pickup" i][name*="time" i]',
  ]);
  out.return_time_raw = readSelect([
    'select#returnTime', 'select#to-time',
    'select[id*="return"][id*="time" i]', 'select[name*="return" i][name*="time" i]',
  ]);
  return out;
}
"""


def _read_enterprise_rental_metadata(page: Page) -> dict:
    try:
        raw = page.evaluate(_ENTERPRISE_METADATA_JS)
    except Exception:
        raw = {}
    out: dict[str, str] = {}
    if raw.get("pickup_date_raw"):
        iso, tm = _parse_rental_field(raw["pickup_date_raw"])
        if iso:
            out["pickup_date"] = iso
        if tm:
            out["pickup_time"] = tm
    if raw.get("return_date_raw"):
        iso, tm = _parse_rental_field(raw["return_date_raw"])
        if iso:
            out["return_date"] = iso
        if tm:
            out["return_time"] = tm
    if raw.get("pickup_time_raw") and "pickup_time" not in out:
        out["pickup_time"] = _normalize_time_str(raw["pickup_time_raw"])
    if raw.get("return_time_raw") and "return_time" not in out:
        out["return_time"] = _normalize_time_str(raw["return_time_raw"])
    return out


def _read_rental_dates_from_page(page: Page) -> dict[str, str]:
    """Backward-compatible wrapper — prefer site-specific readers."""
    meta = _read_sixt_rental_metadata(page)
    meta.update(_read_enterprise_rental_metadata(page))
    return {k: v for k, v in meta.items() if k.endswith("_date")}


def _finalize_rental_metadata(
    page_meta: dict,
    user_meta: dict,
    site: str,
) -> dict:
    """Merge page + user values; fill Sixt/Enterprise site defaults only if still missing."""
    meta = _merge_rental_metadata(page_meta, user_meta)
    if not meta.get("pickup_date") or not meta.get("return_date"):
        if site == "sixt":
            pickup_def, return_def = _sixt_site_default_dates()
        else:
            pickup_def, return_def = _enterprise_site_default_dates()
        meta.setdefault("pickup_date", pickup_def)
        meta.setdefault("return_date", return_def)
    return meta


def _iso_to_us(date_iso: str) -> str:
    """YYYY-MM-DD -> MM/DD/YYYY for Enterprise date inputs."""
    return datetime.strptime(date_iso, "%Y-%m-%d").strftime("%m/%d/%Y")


def _resolve_rental_dates(
    pickup_date: str | None = None,
    return_date: str | None = None,
    pickup_days_ahead: int = 1,
    return_days_ahead: int = 3,
) -> tuple[str, str]:
    """Return pickup/return as YYYY-MM-DD strings."""
    now = now_mst()
    pickup_iso = _to_iso_date(pickup_date) if pickup_date else (
        now + timedelta(days=pickup_days_ahead)
    ).strftime("%Y-%m-%d")
    return_iso = _to_iso_date(return_date) if return_date else (
        now + timedelta(days=return_days_ahead)
    ).strftime("%Y-%m-%d")
    return pickup_iso, return_iso


def _read_input_date(page: Page, selectors: list[str]) -> str | None:
    for sel in selectors:
        try:
            loc = page.locator(sel)
            if loc.count() == 0:
                continue
            el = loc.first
            if not el.is_visible(timeout=800):
                continue
            val = (el.input_value(timeout=800) or el.get_attribute("value") or "").strip()
            if val:
                return _to_iso_date(val)
        except Exception:
            continue
    return None


def _read_rental_dates_from_page(page: Page) -> dict[str, str]:
    """Best-effort read of pickup/return dates from the live search form."""
    pickup = _read_input_date(page, [
        "input#pickupDate", "input#from-date",
        "input[id*='pickup'][id*='date' i]", "input[name*='pickup' i][name*='date' i]",
        "[data-testid='pickup-date']", "[data-testid='pickupDate']",
    ])
    ret = _read_input_date(page, [
        "input#returnDate", "input#to-date",
        "input[id*='return'][id*='date' i]", "input[name*='return' i][name*='date' i]",
        "[data-testid='return-date']", "[data-testid='returnDate']",
    ])
    out: dict[str, str] = {}
    if pickup:
        out["pickup_date"] = pickup
    if ret:
        out["return_date"] = ret
    return out


def _stamp_rental_dates(
    cars: list,
    pickup_date: str,
    return_date: str,
    pickup_time: str = "",
    return_time: str = "",
) -> list:
    meta = {"pickup_date": pickup_date, "return_date": return_date}
    if pickup_time:
        meta["pickup_time"] = pickup_time
    if return_time:
        meta["return_time"] = return_time
    for car in cars:
        car.update(meta)
    return cars


def _ollama_chat(messages, model=EXTRACTOR_MODEL, temperature=0.0, max_tokens=8192, fmt=None):
    payload = {"model": model, "stream": False,
               "options": {"temperature": temperature, "num_predict": max_tokens},
               "messages": messages}
    if fmt is not None:
        payload["format"] = fmt
    resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def _ollama_generate(prompt, model=PLANNER_MODEL, temperature=0.1):
    payload = {"model": model, "stream": False,
               "options": {"temperature": temperature, "num_predict": 4096},
               "prompt": prompt}
    resp = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["response"].strip()


_INTERACTIVE_JS = r"""
() => {
  const sel = 'button, a, input, select, textarea, [role="button"], [role="option"], [role="link"], [onclick], [tabindex]';
  const out = [];
  let idx = 0;
  for (const el of document.querySelectorAll(sel)) {
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) continue;
    const st = window.getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none' || st.opacity === '0') continue;
    const text = (el.innerText || el.value || '').trim().replace(/\s+/g, ' ').slice(0, 90);
    const aria = (el.getAttribute('aria-label') || '').slice(0, 90);
    const ph   = (el.getAttribute('placeholder') || '').slice(0, 90);
    const role = el.getAttribute('role') || '';
    const type = el.getAttribute('type') || '';
    const name = (el.getAttribute('name') || '').slice(0, 60);
    const eid  = (el.id || '').slice(0, 60);
    if (!text && !aria && !ph && !name && !eid) continue;
    el.setAttribute('data-ai-idx', String(idx));
    out.push({idx, tag: el.tagName.toLowerCase(), text, aria, placeholder: ph,
              role, type, name, id: eid,
              x: Math.round(r.left + r.width / 2),
              y: Math.round(r.top + r.height / 2)});
    idx++;
  }
  return out;
}
"""


def _collect_interactive_elements(page, limit: int = 130):
    """Tag and return visible interactive elements (each gets a data-ai-idx attr)."""
    try:
        els = page.evaluate(_INTERACTIVE_JS) or []
        return els[:limit]
    except Exception as e:
        logger.warning(f"  [ai-nav] element collection failed: {e}")
        return []


_PICK_SCHEMA = {
    "type": "object",
    "properties": {
        "index":      {"type": "integer"},
        "confidence": {"type": "number"},
        "reason":     {"type": "string"},
    },
    "required": ["index"],
}


def _ai_pick_element(intent: str, elements: list, model: str = NAV_MODEL):
    """Ask Ollama which interactive element best matches the intent. Returns index or -1."""
    if not elements:
        return -1, 0.0, "no elements"
    compact = [
        {"i": e["idx"], "tag": e["tag"], "text": e.get("text", ""),
         "aria": e.get("aria", ""), "placeholder": e.get("placeholder", ""),
         "role": e.get("role", ""), "type": e.get("type", ""),
         "name": e.get("name", ""), "id": e.get("id", "")}
        for e in elements
    ]
    system = (
        "You are a web UI navigation assistant. You are given a list of interactive "
        "page elements and a target INTENT. Pick the SINGLE element that best fulfils "
        "the intent. Return its 'index'. If none is a good match, return index -1. "
        "Prefer visible buttons/inputs whose text, aria-label, placeholder, name or id "
        "clearly matches the intent."
    )
    user = (
        f"INTENT: {intent}\n\n"
        f"ELEMENTS (JSON):\n{json.dumps(compact, ensure_ascii=False)}\n\n"
        "Return JSON: {\"index\": <int>, \"confidence\": <0..1>, \"reason\": \"...\"}"
    )
    try:
        raw = _ollama_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model, temperature=0.0, fmt=_PICK_SCHEMA,
        )
        obj = json.loads(raw)
        return int(obj.get("index", -1)), float(obj.get("confidence", 0.0)), obj.get("reason", "")
    except Exception as e:
        logger.warning(f"  [ai-nav] pick failed: {e}")
        return -1, 0.0, str(e)


def _ai_vision_click(page, intent: str, log_fn=print, model: str = VISION_MODEL) -> bool:
    """Last-resort: screenshot the page and ask a vision model for click coordinates."""
    import base64
    try:
        png = page.screenshot()
        b64 = base64.b64encode(png).decode()
        vp = page.viewport_size or {"width": 1280, "height": 800}
        system = (
            "You locate UI elements in screenshots. Return ONLY pixel coordinates of the "
            "CENTER of the element matching the intent, as JSON {\"x\": int, \"y\": int}. "
            f"Image is {vp['width']}x{vp['height']} pixels."
        )
        raw = _ollama_chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": f"INTENT: {intent}", "images": [b64]}],
            model=model, temperature=0.0,
            fmt={"type": "object",
                 "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                 "required": ["x", "y"]},
        )
        obj = json.loads(raw)
        x, y = int(obj["x"]), int(obj["y"])
        page.mouse.click(x, y)
        log_fn(f"   [ai-vision] clicked ({x},{y}) for: {intent}")
        page.wait_for_timeout(1500)
        return True
    except Exception as e:
        log_fn(f"   [ai-vision] failed: {e}")
        return False


def _click_selectors(page, selectors, log_fn) -> bool:
    for sel in selectors:
        try:
            loc = page.locator(sel)
            if loc.count() > 0 and loc.first.is_visible(timeout=1200):
                loc.first.scroll_into_view_if_needed()
                loc.first.click(timeout=4000)
                log_fn(f"   [click] selector: {sel}")
                return True
        except Exception:
            continue
    return False


def _click_ai(page, intent, min_confidence, log_fn) -> bool:
    elements = _collect_interactive_elements(page)
    idx, conf, reason = _ai_pick_element(intent, elements)
    if idx >= 0 and conf >= min_confidence:
        try:
            target = page.locator(f"[data-ai-idx='{idx}']")
            if target.count() > 0:
                target.first.scroll_into_view_if_needed()
                target.first.click(timeout=4000)
                log_fn(f"   [click] AI picked idx={idx} (conf={conf:.2f}): {reason}")
                return True
        except Exception as e:
            log_fn(f"   [click] AI element click failed: {e}")
    return False


def _smart_click(page, intent: str, selectors=None, log_fn=print,
                 min_confidence: float = None, use_vision: bool = False) -> bool:
    """Click an element by intent.

    AI_FIRST=False (default): try hardcoded selectors, then AI, then vision.
    AI_FIRST=True: ask Ollama first (and vision), and only fall back to selectors
    if the AI couldn't satisfy the intent.
    """
    selectors = selectors or []
    conf = AI_PICK_CONFIDENCE if min_confidence is None else min_confidence

    if AI_FIRST:
        log_fn(f"   [click] AI-first for: {intent}")
        if _click_ai(page, intent, conf, log_fn):
            return True
        if _ai_vision_click(page, intent, log_fn):
            return True
        if _click_selectors(page, selectors, log_fn):
            return True
        log_fn(f"   [click] could not satisfy intent: {intent}")
        return False

    if _click_selectors(page, selectors, log_fn):
        return True
    log_fn(f"   [click] selectors failed — asking AI for: {intent}")
    if _click_ai(page, intent, conf, log_fn):
        return True
    if use_vision:
        log_fn(f"   [click] trying vision fallback for: {intent}")
        return _ai_vision_click(page, intent, log_fn)
    log_fn(f"   [click] could not satisfy intent: {intent}")
    return False


def _fill_selectors(page, selectors, value, log_fn) -> bool:
    for sel in selectors:
        try:
            loc = page.locator(sel)
            if loc.count() > 0 and loc.first.is_visible(timeout=1200):
                loc.first.click()
                loc.first.fill("")
                page.wait_for_timeout(150)
                loc.first.fill(value)
                log_fn(f"   [fill] selector: {sel} = {value}")
                return True
        except Exception:
            continue
    return False


def _fill_ai(page, intent, value, min_confidence, log_fn) -> bool:
    elements = _collect_interactive_elements(page)
    inputs = [e for e in elements if e["tag"] in ("input", "textarea")] or elements
    idx, conf, reason = _ai_pick_element(intent, inputs)
    if idx >= 0 and conf >= min_confidence:
        try:
            target = page.locator(f"[data-ai-idx='{idx}']")
            if target.count() > 0:
                target.first.click()
                target.first.fill("")
                page.wait_for_timeout(150)
                target.first.fill(value)
                log_fn(f"   [fill] AI picked idx={idx} (conf={conf:.2f}) = {value}")
                return True
        except Exception as e:
            log_fn(f"   [fill] AI element fill failed: {e}")
    return False


def _smart_fill(page, intent: str, value: str, selectors=None, log_fn=print,
                min_confidence: float = None) -> bool:
    """Fill an input by intent. AI-first when AI_FIRST, else selectors-first."""
    selectors = selectors or []
    conf = AI_PICK_CONFIDENCE if min_confidence is None else min_confidence

    if AI_FIRST:
        log_fn(f"   [fill] AI-first for: {intent}")
        if _fill_ai(page, intent, value, conf, log_fn):
            return True
        if _fill_selectors(page, selectors, value, log_fn):
            return True
        log_fn(f"   [fill] could not satisfy intent: {intent}")
        return False

    if _fill_selectors(page, selectors, value, log_fn):
        return True
    log_fn(f"   [fill] selectors failed — asking AI for: {intent}")
    if _fill_ai(page, intent, value, conf, log_fn):
        return True
    log_fn(f"   [fill] could not satisfy intent: {intent}")
    return False


def _parse_json_array(raw):
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(raw[start:end + 1])
        except Exception as e:
            logger.warning(f"JSON parse failed: {e}")
    return []

@lru_cache(maxsize=256)
def resolve_iata(code: str) -> str:
    """Turn a 3-letter IATA code (e.g. 'YYC') into its airport name.

    Anything that isn't a 3-letter code is returned unchanged, so it's safe to
    pass a full city/airport name straight through. Results are cached so the
    airport database is only downloaded once per code.
    """
    code = (code or "").strip()
    if not (len(code) == 3 and code.isalpha()):
        return code
    try:
        url = ("https://raw.githubusercontent.com/jpatokal/openflights/"
               "master/data/airports.dat")
        df = pd.read_csv(url, header=None,
                         names=["id","name","city","country","iata","icao",
                                "lat","lon","alt","tz","dst","tzdb","type","source"])
        row = df[df["iata"].str.upper() == code.upper()]
        if not row.empty:
            return str(row.iloc[0]["name"])
    except Exception as e:
        logger.warning(f"resolve_iata: {e}")
    return code

resolve_iata_code = resolve_iata


PRICE_RE = re.compile(r"((?:CA)?\s*\$\s*[\d,]+\.?\d*\s*/\s*day)", re.IGNORECASE)
CAR_TYPES = [
    "Economy","Compact","Intermediate","Standard","Fullsize","Full-Size",
    "Premium","Luxury","SUV","Minivan","Van","Convertible","Pickup",
    "Wagon","Crossover","Electric","Hybrid","Sedan","Elite","Truck",
]
TYPE_RE   = re.compile(r"\b(" + "|".join(CAR_TYPES) + r")\b", re.IGNORECASE)
TRANS_RE  = re.compile(r"\b(Automatic|Manual)\b", re.IGNORECASE)
MODEL_PAREN_RE = re.compile(r"\(([^)]{4,60})\)")
SIXT_CLASS_LINE_RE = re.compile(
    r"^(ECONOMY|COMPACT|INTERMEDIATE|STANDARD|FULLSIZE|FULL-SIZE|PREMIUM|MINIVAN|LUXURY|VAN|SUV|CONVERTIBLE)"
    r"(\s+ELITE)?\s+\(([^)]+)\)\s*$",
    re.IGNORECASE,
)
OR_SIMILAR_RE = re.compile(r"^(.+?)\s+or\s+similar\s*$", re.IGNORECASE)
PEOPLE_RE = re.compile(r"(\d+)\s+People", re.IGNORECASE)
BAGS_RE = re.compile(r"(\d+)\s+Bags", re.IGNORECASE)
ENTERPRISE_CARD_RE = re.compile(
    r"(?P<category>[A-Za-z0-9][A-Za-z0-9 \-]+?)\s*\n+"
    r"(?P<model>[^\n]+? or similar)\s*\n+Automatic\s*\n+"
    r"(?P<seats>\d+)\s+People\s*\n+"
    r"(?P<bags>\d+)\s+Bags\s*\n+"
    # Bounded, line-anchored gap. Using [^\n] (no DOTALL) keeps each line
    # match unambiguous and the {0,20} cap prevents the catastrophic
    # backtracking that froze parsing on cards with no "Per Day" price
    # (e.g. the trailing "Call For Availability" classes).
    r"(?:[^\n]*\n){0,20}?"
    r"(?P<per_day>\$[\d,]+\.?\d*)\s*\n+Per Day",
    re.IGNORECASE | re.MULTILINE,
)

def _enterprise_results_present(page: Page) -> bool:
    """Return true once the results page has at least one vehicle offer visible."""
    try:
        body = page.inner_text("body", timeout=1500)
    except Exception:
        return False
    return "Per Day" in body or "Call For Availability" in body


def _scroll_enterprise_results(page: Page, max_scrolls: int = 8) -> None:
    """Scroll enough to hydrate lazy-loaded Enterprise cards without a long blind wait."""
    stable_rounds = 0
    last_height = None
    for _ in range(max_scrolls):
        try:
            height = page.evaluate("document.body.scrollHeight")
            if height == last_height:
                stable_rounds += 1
            else:
                stable_rounds = 0
            last_height = height

            if stable_rounds >= 2 and _enterprise_results_present(page):
                break

            page.evaluate("window.scrollBy({top: 650, left: 0, behavior: 'auto'})")
            page.wait_for_timeout(300)
        except Exception:
            break
TITLE_NAME_RE  = re.compile(
    r"^[A-Z][a-zA-Z\-]+(?:\s[A-Z][a-zA-Z0-9\-]+){1,4}(?:\s\(or similar\))?$")
FALSE_POS = {"Automatic","Manual","Standard","Select","Filter","Sort","Price",
             "Day","Per","Choose","Search","Rated","Pick","View","All Cars",
             "Recommended","Featured","Pay Later","Unlimited","Kilometers"}
NUM_RE = re.compile(r"[\d,]+\.?\d*")


def _normalise_price(raw):
    p = re.sub(r"\s*/\s*", "/", re.sub(r"\s+", "", raw))
    if not p.upper().startswith("CA"):
        p = "CA" + p
    return p


def _usd_to_ca_per_day(usd_str: str) -> str:
    m = re.search(r"([\d,]+\.?\d*)", usd_str.replace("$", ""))
    if not m:
        return ""
    val = m.group(1).replace(",", "")
    return f"CA${val}/day"


def _normalize_enterprise_text(page_text: str) -> str:
    """Join Enterprise split-dollar lines ($ / 143 / .83) into $143.83."""
    lines = page_text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == "$" and i + 2 < len(lines):
            whole, frac = lines[i + 1].strip(), lines[i + 2].strip()
            if re.match(r"^\d+$", whole) and re.match(r"^\.\d+$", frac):
                out.append(f"${whole}{frac}")
                i += 3
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _parse_seats_bags_from_lines(lines: list[str]) -> tuple:
    """Parse Sixt-style standalone seat/bag lines before price."""
    seats = bags = None
    stop = len(lines)
    for j, w in enumerate(lines):
        low = w.lower()
        if low.startswith("unlimited"):
            stop = j
            break
        if PRICE_RE.search(w):
            stop = j
            break
    nums = []
    for w in lines[:stop]:
        sm = re.match(r"^(\d+)\s*(?:\(\d+\+\d+\))?\s*$", w)
        if sm:
            n = int(sm.group(1))
            if 1 <= n <= 9:
                nums.append(n)
    if len(nums) >= 2:
        seats, bags = nums[-2], nums[-1]
    elif len(nums) == 1:
        seats = nums[0]
    return seats, bags


_SIXT_TYPE_SET = {t.upper() for t in CAR_TYPES}
_SIXT_MODEL_LINE_RE = re.compile(r"^[A-Z0-9][A-Z0-9 \-]{2,45}$")


def _sixt_type_rank(type_up: str) -> int:
    """Prefer SUV/Premium over generic 'Standard' when both match a model line."""
    if type_up in ("SUV", "MINIVAN", "VAN", "PICKUP", "TRUCK", "CROSSOVER"):
        return 3
    if type_up in ("PREMIUM", "LUXURY", "FULLSIZE", "FULL-SIZE", "INTERMEDIATE", "COMPACT"):
        return 2
    if type_up == "STANDARD":
        return 1
    return 2


def _resolve_sixt_car_header(window: list[str]) -> tuple[str, str, list[str]]:
    """Return (car_name, car_type, header_slice) from lines above a Sixt price."""
    car_name, car_type = "Unknown", "Unknown"
    header_slice = window

    for j, w in enumerate(window):
        hm = SIXT_CLASS_LINE_RE.match(w)
        if hm:
            base_type = hm.group(1).replace("-", "").title()
            if base_type.lower() == "fullsize":
                base_type = "Fullsize"
            car_type = f"{base_type} Elite" if hm.group(2) else base_type
            car_name = hm.group(3).strip()
            return car_name, car_type, window[j:]

    # Sixt.ca often lists class and model on separate lines (e.g. SUV / GMC ACADIA).
    best_split = None
    for j, w in enumerate(window):
        w_st = w.strip()
        w_up = w_st.upper()
        if w_up not in _SIXT_TYPE_SET:
            continue
        if w_up == "STANDARD" and j > 0 and window[j - 1].strip().lower() in (
            "select", "filter", "sort", "recommended",
        ):
            continue
        type_label = "Fullsize" if w_up.replace("-", "") == "FULLSIZE" else w_st.title()
        for k in range(j + 1, min(j + 5, len(window))):
            nxt = window[k].strip()
            if not nxt or nxt.lower() in ("automatic", "manual"):
                continue
            if PRICE_RE.search(nxt):
                break
            if _SIXT_MODEL_LINE_RE.match(nxt) and len(nxt.split()) >= 2:
                if nxt.upper() not in _SIXT_TYPE_SET and nxt not in FALSE_POS:
                    rank = _sixt_type_rank(w_up)
                    if best_split is None or (k, rank) > (best_split[0], best_split[1]):
                        best_split = (k, rank, nxt, type_label, window[j:])
    if best_split:
        _, _, car_name, car_type, header_slice = best_split
        return car_name, car_type, header_slice

    for w in reversed(window):
        w_st = w.strip()
        if w_st in FALSE_POS or w_st.lower() in ("automatic", "manual", "unlimited"):
            continue
        if PRICE_RE.search(w_st):
            continue
        if _SIXT_MODEL_LINE_RE.match(w_st) and len(w_st.split()) >= 2:
            if w_st.upper() not in _SIXT_TYPE_SET:
                car_name = w_st
                break
        pm = MODEL_PAREN_RE.search(w_st)
        if pm and car_name == "Unknown":
            candidate = pm.group(1).strip()
            if candidate.lower() not in {"or similar", "awd", "4wd", "4x4"}:
                car_name = candidate
        if car_name == "Unknown" and TITLE_NAME_RE.match(w_st) and len(w_st.split()) >= 2:
            car_name = w_st

    if car_type == "Unknown":
        for w in reversed(window):
            tm = TYPE_RE.search(w)
            if tm:
                car_type = tm.group(1).title()
                break
            w_up = w.strip().upper()
            if w_up in _SIXT_TYPE_SET:
                car_type = "Fullsize" if w_up.replace("-", "") == "FULLSIZE" else w.strip().title()
                break

    return car_name, car_type, header_slice


def _extract_sixt_cards(page_text: str, location: str) -> list[dict]:
    """Deterministic Sixt offer parser — no AI."""
    lines = [l.strip() for l in page_text.split("\n") if l.strip()]
    cars = []
    last_price_idx = -1
    for i, line in enumerate(lines):
        price_m = PRICE_RE.search(line)
        if not price_m:
            continue
        raw_price = _normalise_price(price_m.group(1))
        num_m = NUM_RE.search(raw_price)
        if not num_m:
            continue
        try:
            val = float(num_m.group().replace(",", ""))
        except ValueError:
            continue
        if not (10.0 <= val <= 5000.0):
            continue

        window = lines[last_price_idx + 1:i]
        last_price_idx = i
        car_name, car_type, header_slice = _resolve_sixt_car_header(window)
        transmission = "Automatic"
        seats, bags = _parse_seats_bags_from_lines(header_slice)
        for w in reversed(window):
            if re.search(r"\b(automatic|manual)\b", w, re.I):
                tr = TRANS_RE.search(w)
                if tr:
                    transmission = tr.group(1).title()
                break

        if car_name == "Unknown":
            continue

        cars.append({
            "car_name": car_name,
            "car_type": car_type,
            "price_per_day": raw_price,
            "transmission": transmission,
            "seats": seats,
            "bags": bags,
            "location": location,
            "_source": "sixt_parser",
        })

    seen, unique = set(), []
    for c in cars:
        key = (c["car_name"].lower(), c["price_per_day"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(c)
    logger.info(f"  [sixt_parser] {len(unique)} cars")
    return unique


def _truncate_enterprise_listing(page_text: str) -> str:
    """Keep only featured vehicles — stop before Explore/Hide Alternative section."""
    low = page_text.lower()
    markers = (
        "hide alternative",
        "explore alternative",
        "explore alternative possibilities",
    )
    cut = len(page_text)
    for marker in markers:
        idx = low.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    if cut < len(page_text):
        logger.info("  [enterprise] truncated listing before alternative vehicles section")
    return page_text[:cut]


def _extract_enterprise_cards(page_text: str, location: str) -> list[dict]:
    """Deterministic Enterprise vehicle-class parser — real model names + per-day price."""
    text = _normalize_enterprise_text(_truncate_enterprise_listing(page_text))
    cars = []
    for m in ENTERPRISE_CARD_RE.finditer(text):
        category = m.group("category").strip()
        model = m.group("model").strip()
        if category.lower() in FALSE_POS or "recommended" in category.lower():
            continue
        if "explore alternative" in model.lower():
            continue
        per_day = _usd_to_ca_per_day(m.group("per_day"))
        if not per_day:
            continue
        cars.append({
            "car_name": model,
            "car_type": category,
            "price_per_day": per_day,
            "transmission": "Automatic",
            "seats": int(m.group("seats")),
            "bags": int(m.group("bags")),
            "location": location,
            "_source": "enterprise_parser",
        })

    seen, unique = set(), []
    for c in cars:
        key = (c["car_name"].lower(), c["price_per_day"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(c)
    logger.info(f"  [enterprise_parser] {len(unique)} cars")
    return unique


def _extract_dom_sixt(page: Page, location: str) -> list[dict]:
    """Pull Sixt offer card text from the live DOM."""
    try:
        chunks = page.evaluate("""
            () => {
                const out = [];
                const seen = new Set();
                const nodes = document.querySelectorAll(
                    '[class*="offer"], [class*="vehicle"], [class*="car"], [data-testid*="offer"], article'
                );
                for (const el of nodes) {
                    const t = (el.innerText || '').trim();
                    if (!t || t.length < 40 || t.length > 2500) continue;
                    if (!/CA\\$[\\d,.]+\\s*\\/\\s*day/i.test(t)) continue;
                    const key = t.slice(0, 120);
                    if (seen.has(key)) continue;
                    seen.add(key);
                    out.push(t);
                }
                return out;
            }
        """)
        merged = []
        for chunk in chunks or []:
            merged.extend(_extract_sixt_cards(chunk, location))
        logger.info(f"  [dom_sixt] {len(merged)} cars")
        return merged
    except Exception as e:
        logger.warning(f"  [dom_sixt] failed: {e}")
        return []


def _extract_dom_enterprise(page: Page, location: str) -> list[dict]:
    """Pull Enterprise vehicle cards from the live DOM."""
    try:
        chunks = page.evaluate("""
            () => {
                const out = [];
                const seen = new Set();
                for (const el of document.querySelectorAll(
                    '[class*="vehicle"], [class*="Vehicle"], article, [data-testid*="vehicle"]'
                )) {
                    const t = (el.innerText || '').trim();
                    if (!t || t.length < 50 || t.length > 2000) continue;
                    if (!/or similar/i.test(t)) continue;
                    if (!/People/i.test(t) || !/Per Day/i.test(t)) continue;
                    const key = t.slice(0, 100);
                    if (seen.has(key)) continue;
                    seen.add(key);
                    out.push(t);
                }
                return out;
            }
        """)
        merged = []
        for chunk in chunks or []:
            merged.extend(_extract_enterprise_cards(chunk, location))
        logger.info(f"  [dom_enterprise] {len(merged)} cars")
        return merged
    except Exception as e:
        logger.warning(f"  [dom_enterprise] failed: {e}")
        return []


def _regex_extract(page_text, location):
    lines = [l.strip() for l in page_text.split("\n") if l.strip()]
    cars  = []
    for idx, line in enumerate(lines):
        price_m = PRICE_RE.search(line)
        if not price_m:
            continue
        raw_price = _normalise_price(price_m.group(1).strip())
        num_m = NUM_RE.search(raw_price)
        if not num_m:
            continue
        try:
            val = float(num_m.group().replace(",", ""))
        except ValueError:
            continue
        if not (10.0 <= val <= 1500.0):
            continue
        window = lines[max(0, idx - 30):idx]
        car_name, car_type, transmission = "Unknown", "Unknown", "Automatic"
        seats, bags = None, None
        for w in reversed(window):
            pm = MODEL_PAREN_RE.search(w)
            if pm and car_name == "Unknown":
                candidate = pm.group(1).strip()
                if candidate.lower() not in {"or similar","awd","4wd","4x4"}:
                    car_name = candidate
                    continue
            if (car_name == "Unknown" and TITLE_NAME_RE.match(w)
                    and w not in FALSE_POS and len(w.split()) >= 2):
                car_name = w.strip()
            if car_type == "Unknown":
                tm = TYPE_RE.search(w)
                if tm:
                    car_type = tm.group(1).title()
            trm = TRANS_RE.search(w)
            if trm:
                transmission = trm.group(1).title()
            if seats is None:
                if re.match(r"^[2-9]$", w):
                    seats = int(w)
                else:
                    sm = re.search(r"\b([2-9])\s*(?:seat|passenger)", w, re.I)
                    if sm:
                        seats = int(sm.group(1))
            if bags is None:
                bm = re.search(r"\b([1-6])\s*(?:bag|luggage|suitcase)", w, re.I)
                if bm:
                    bags = int(bm.group(1))
            nums = re.findall(r"\b[1-6]\b", " ".join(window))
            if seats is None and len(nums) >= 1:
                seats = int(nums[0])
            if bags is None and len(nums) >= 2:
                bags = int(nums[1])
        if car_name != "Unknown" or car_type != "Unknown":
            cars.append({"car_name": car_name, "car_type": car_type,
                         "price_per_day": raw_price, "transmission": transmission,
                         "seats": seats, "bags": bags, "location": location})
    seen, unique = set(), []
    for c in cars:
        key = (c["car_name"].lower().strip(), c["price_per_day"].lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(c)
    logger.info(f"  [regex] {len(unique)} unique cars found")
    return unique


def _ollama_enrich(cars, page_text, location, site_hint=""):
    """Only fill Unknown fields — never overwrite parser/DOM values."""
    needs = [c for c in cars
             if c.get("car_name") in (None, "", "Unknown")
             or c.get("car_type") in (None, "", "Unknown")]
    if not needs:
        return cars
    system = (
        "You are a car-rental data enrichment engine.\n"
        "ONLY fill car_name / car_type where value is 'Unknown'.\n"
        "NEVER change price_per_day, transmission, seats, bags, location.\n"
        "NEVER invent cars. Output ONLY the updated JSON array."
    )
    user = (
        f"Site: {site_hint or location}\n\nRecords:\n{json.dumps(needs, indent=2)}\n\n"
        f"Page text:\n{page_text[:18000]}\n\n"
        f"Return EXACTLY {len(needs)} records in SAME ORDER. ONLY valid JSON array."
    )
    try:
        raw = _ollama_chat([{"role": "system", "content": system},
                            {"role": "user", "content": user}])
        enriched = _parse_json_array(raw)
        if len(enriched) != len(needs):
            logger.warning(f"  [ollama] count mismatch ({len(enriched)} vs {len(needs)})")
            return cars
        by_key = {(c["car_name"], c["price_per_day"]): c for c in enriched}
        for orig in cars:
            key = (orig.get("car_name"), orig.get("price_per_day"))
            enr = by_key.get(key)
            if not enr:
                continue
            if orig.get("car_name") in (None, "", "Unknown") and enr.get("car_name"):
                orig["car_name"] = enr["car_name"]
            if orig.get("car_type") in (None, "", "Unknown") and enr.get("car_type"):
                orig["car_type"] = enr["car_type"]
        logger.info(f"  [ollama] enriched {len(enriched)} unknown fields")
    except Exception as e:
        logger.warning(f"  [ollama] enrichment failed: {e}")
    return cars


def _sanity_filter(cars):
    good = []
    for c in cars:
        m = NUM_RE.search(str(c.get("price_per_day", "")))
        if not m:
            logger.warning(f"  [filter] no number — dropping")
            continue
        try:
            val = float(m.group().replace(",",""))
        except ValueError:
            continue
        if 10.0 <= val <= 1500.0:
            good.append(c)
        else:
            logger.warning(f"  [filter] {val} out of range — dropping {c.get('car_name')}")
    return good


def _extract_dom_blocks(page: Page, location: str):
    """
    Extract structured car cards using DOM grouping (VERY IMPORTANT for modern JS sites)
    """
    cars = []

    try:
        cards = page.locator("[class*='car'], [class*='vehicle'], [data-testid*='car']").all()

        for card in cards:
            try:
                text = card.inner_text(timeout=500)

                price_m = PRICE_RE.search(text)
                if not price_m:
                    continue

                price = _normalise_price(price_m.group(1))

                # Extract attributes inside card
                car_name = "Unknown"
                car_type = "Unknown"
                transmission = "Automatic"
                seats = None
                bags = None

                lines = [l.strip() for l in text.split("\n") if l.strip()]

                for l in lines:
                    if car_name == "Unknown" and TITLE_NAME_RE.match(l):
                        car_name = l

                    if car_type == "Unknown":
                        tm = TYPE_RE.search(l)
                        if tm:
                            car_type = tm.group(1).title()

                    trm = TRANS_RE.search(l)
                    if trm:
                        transmission = trm.group(1).title()

                    sm = re.search(r"\b([2-9])\s*(seat|passenger)", l, re.I)
                    if sm:
                        seats = int(sm.group(1))

                    bm = re.search(r"\b([1-6])\s*(bag|luggage)", l, re.I)
                    if bm:
                        bags = int(bm.group(1))

                cars.append({
                    "car_name": car_name,
                    "car_type": car_type,
                    "price_per_day": price,
                    "transmission": transmission,
                    "seats": seats,
                    "bags": bags,
                    "location": location,
                    "_source": "dom"
                })

            except Exception:
                continue

    except Exception:
        pass

    logger.info(f"  [dom] {len(cars)} cars")
    return cars



def _regex_extract_v2(page_text, location):
    """Backward-compatible alias — uses the Sixt block parser."""
    return _extract_sixt_cards(page_text, location)


def _source_score(source: str) -> int:
    weights = {
        "sixt_parser": 10,
        "enterprise_parser": 10,
        "dom_sixt": 8,
        "dom_enterprise": 8,
        "dom": 5,
        "regex": 4,
        "ollama": 1,
    }
    return weights.get(source or "", 0)


def _merge_record(existing: dict, incoming: dict) -> dict:
    """Merge two records with the same key — keep best fields, no hallucination."""
    ex_score = _source_score(existing.get("_source"))
    in_score = _source_score(incoming.get("_source"))
    base, extra = (existing, incoming) if ex_score >= in_score else (incoming, existing)
    out = dict(base)
    for field in ("car_name", "car_type", "price_per_day", "transmission",
                  "seats", "bags", "location", "_source"):
        bval, xval = base.get(field), extra.get(field)
        if field in ("seats", "bags"):
            if out.get(field) is None and xval is not None:
                out[field] = xval
        elif field == "car_name":
            if out.get(field) in (None, "", "Unknown") and xval not in (None, "", "Unknown"):
                out[field] = xval
        elif field == "car_type":
            if out.get(field) in (None, "", "Unknown") and xval not in (None, "", "Unknown"):
                out[field] = xval
        elif field == "location":
            if not out.get(field) and xval:
                out[field] = xval
        elif field == "_source":
            out["_source"] = base.get("_source") or extra.get("_source")
    out["_confidence"] = _source_score(out.get("_source"))
    if out.get("car_name") not in (None, "", "Unknown"):
        out["_confidence"] += 1
    if out.get("seats") is not None:
        out["_confidence"] += 1
    if out.get("bags") is not None:
        out["_confidence"] += 1
    return out


def _norm_name_key(name) -> str:
    """Normalize a car name for de-duplication.

    Ollama tends to keep a suffix (e.g. "Mazda CX-5 or similar" or
    "Mazda CX-5 or similar model") while the deterministic parser strips it
    ("Mazda CX-5"). Collapsing both to the same key stops AI-merge from listing
    every car twice.
    """
    s = str(name or "").lower().strip()
    s = re.sub(r"\s+or\s+similar(?:\s+model)?\s*$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_price_key(price) -> str:
    """Normalize a price so 'CA$149.89/day' and '$149.89/day' match."""
    return re.sub(r"[^\d.]", "", str(price or "")).strip(".")


def _merge_sixt_by_price(*sources) -> list[dict]:
    """Merge by price: earlier sources win; later sources fill empty fields only."""
    by_price: dict[str, dict] = {}
    for source in sources:
        for c in source:
            pk = _norm_price_key(c.get("price_per_day", ""))
            if not pk:
                continue
            if pk not in by_price:
                by_price[pk] = dict(c)
            else:
                base = by_price[pk]
                for field in ("seats", "bags", "transmission", "car_type", "car_name"):
                    if base.get(field) in (None, "", "Unknown") and c.get(field) not in (None, "", "Unknown"):
                        base[field] = c[field]
    return list(by_price.values())


def _backfill_from_parser(ai_cars: list[dict], parser_cars: list[dict]) -> list[dict]:
    """Keep AI row count; fill empty fields from parser matched by price."""
    if not ai_cars:
        return parser_cars
    if not parser_cars:
        return ai_cars
    by_price = {
        _norm_price_key(c.get("price_per_day", "")): c
        for c in parser_cars
        if _norm_price_key(c.get("price_per_day", ""))
    }
    for car in ai_cars:
        parser = by_price.get(_norm_price_key(car.get("price_per_day", "")))
        if not parser:
            continue
        for field in ("seats", "bags", "transmission", "car_type", "car_name"):
            if car.get(field) in (None, "", "Unknown") and parser.get(field) not in (None, "", "Unknown"):
                car[field] = parser[field]
    return ai_cars


def _sixt_text_for_ollama(page_text: str) -> str:
    """Compact card-focused text so Ollama sees every offer, not page chrome."""
    lines = [l.strip() for l in page_text.split("\n") if l.strip()]
    chunks, seen_prices = [], set()
    for i, line in enumerate(lines):
        if not re.search(r"CA\$[\d,.]+/day", line, re.I):
            continue
        pk = re.sub(r"[^\d.]", "", line)
        if pk in seen_prices:
            continue
        seen_prices.add(pk)
        chunks.append("\n".join(lines[max(0, i - 10): i + 1]))
    if chunks:
        return "\n---\n".join(chunks)
    return page_text[:30000]


def _ollama_extract_prompts(location: str, site_hint: str, page_text: str) -> tuple[str, str]:
    """Site-specific strict prompts for Ollama extraction."""
    hint = (site_hint or "").lower()
    is_sixt = "sixt" in hint
    is_enterprise = "enterprise" in hint
    text_for_ai = _sixt_text_for_ollama(page_text) if is_sixt else page_text[:30000]

    system = (
        "You are a STRICT car-rental JSON extraction engine.\n"
        "GLOBAL RULES:\n"
        "- ONLY extract vehicles that clearly exist in the page text\n"
        "- NEVER guess, infer, or hallucinate cars or prices\n"
        "- price_per_day MUST be CA$XX.XX/day (include CA$ and /day)\n"
        "- SKIP 'Call For Availability' rows without a visible per-day price\n"
        "- Output ONLY a valid JSON array — no markdown, no commentary\n"
    )

    if is_sixt:
        system += (
            "\nSIXT.CA — extract EVERY offer that has CA$/day (typically 10–15 cars).\n"
            "Each card block looks like:\n"
            "  1) CLASS line (e.g. STANDARD SUV, COMPACT SEDAN, Premium, STANDARD SUV)\n"
            "  2) MODEL line in ALL CAPS (e.g. GMC ACADIA, VOLKSWAGEN JETTA)\n"
            "  3) Or similar model (ignore this line)\n"
            "  4) two numbers = seats and bags (e.g. 7 then 4, or 5 then 3)\n"
            "  5) Automatic or Manual\n"
            "  6) CA$61.91/day (use this exact price line, ignore CA$XXX.XXtotal lines)\n"
            "\nSIXT FIELD RULES:\n"
            "- car_name: ALL-CAPS model line only (GMC ACADIA, BMW 3 SERIES)\n"
            "- car_type: the class line above the model — use as shown (Standard Suv, Compact Sedan, Premium)\n"
            "- transmission: Automatic or Manual from the card; null only if truly absent\n"
            "- seats/bags: from the two numbers before Automatic; null only if truly absent\n"
            "- DO NOT skip a car just because seats/bags are unclear — include name + price\n"
            "- Ignore filters, sidebar, member-rate promos, and 'total' prices\n"
            "- One JSON object per unique CA$/day price\n"
        )
        example = (
            '"car_name": "GMC ACADIA",\n'
            '    "car_type": "Standard Suv",\n'
            '    "price_per_day": "CA$74.38/day",\n'
            '    "transmission": "Automatic",\n'
            '    "seats": 7,\n'
            '    "bags": 4'
        )
    elif is_enterprise:
        system += (
            "- transmission MUST be 'Automatic' or 'Manual' when shown — null if not visible\n"
            "- seats and bags MUST be integers when 'X People' / 'X Bags' are shown\n"
            "- If a required field is missing on a card → SKIP that vehicle\n"
        )
        system += (
            "\nENTERPRISE.COM — each offer card contains:\n"
            "  1) vehicle class line (e.g. Compact SUV, Midsize, Premium)\n"
            "  2) model line ending in 'or similar' (e.g. Nissan Kicks or similar)\n"
            "  3) Automatic\n"
            "  4) X People, X Bags\n"
            "  5) $XX.XX then 'Per Day'\n"
            "\nENTERPRISE FIELD RULES:\n"
            "- car_name: model line with 'or similar' suffix\n"
            "- car_type: vehicle class line above the model\n"
            "- price_per_day: convert $XX.XX Per Day to CA$XX.XX/day\n"
            "- transmission: almost always Automatic\n"
            "- seats/bags: from 'X People' and 'X Bags' lines\n"
            "- Skip 'Explore Alternative' / hidden vehicles section\n"
        )
        example = (
            '"car_name": "Nissan Kicks or similar",\n'
            '    "car_type": "Compact SUV",\n'
            '    "price_per_day": "CA$78.84/day",\n'
            '    "transmission": "Automatic",\n'
            '    "seats": 5,\n'
            '    "bags": 3'
        )
    else:
        example = (
            '"car_name": "...",\n'
            '    "car_type": "...",\n'
            '    "price_per_day": "CA$00.00/day",\n'
            '    "transmission": "Automatic",\n'
            '    "seats": 5,\n'
            '    "bags": 3'
        )

    user = f"""Extract every rental offer from this page.

Location: {location}
Site: {site_hint}

Every object MUST include transmission, seats, and bags when they appear on the card.
Return JSON array:
[
  {{
    {example},
    "location": "{location}"
  }}
]

PAGE TEXT:
{text_for_ai}
"""
    return system, user


def _ollama_strict_extract(page_text, location, site_hint=""):
    system, user = _ollama_extract_prompts(location, site_hint, page_text)

    try:
        raw = _ollama_chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])

        cars = _parse_json_array(raw)
        for c in cars:
            c["_source"] = "ollama"
            if not c.get("location"):
                c["location"] = location
            if c.get("price_per_day"):
                c["price_per_day"] = _normalise_price(str(c["price_per_day"]))
            if c.get("transmission"):
                c["transmission"] = str(c["transmission"]).title()
            for field in ("seats", "bags"):
                if c.get(field) is not None:
                    try:
                        c[field] = int(float(c[field]))
                    except (TypeError, ValueError):
                        c[field] = None

        logger.info(f"  [ollama_strict] {len(cars)} cars")
        return cars

    except Exception as e:
        logger.warning(f"ollama strict failed: {e}")
        return []


def _ollama_script_prompt_src() -> str:
    """Embed the same Ollama prompt helpers used by run.py into generated scripts."""
    import inspect
    src = inspect.getsource(_ollama_extract_prompts) + "\n" + inspect.getsource(_ollama_strict_extract)
    return src.replace("logger.info(", "log(").replace("logger.warning(", "log(")


def _merge_results(*sources):
    by_key: dict[tuple, dict] = {}
    for source in sources:
        for c in source:
            if not c.get("price_per_day"):
                continue
            key = (
                _norm_name_key(c.get("car_name", "")),
                _norm_price_key(c.get("price_per_day", "")),
            )
            if key in by_key:
                by_key[key] = _merge_record(by_key[key], c)
            else:
                entry = dict(c)
                entry["_confidence"] = _source_score(entry.get("_source"))
                if entry.get("car_name") not in (None, "", "Unknown"):
                    entry["_confidence"] += 1
                if entry.get("seats") is not None:
                    entry["_confidence"] += 1
                if entry.get("bags") is not None:
                    entry["_confidence"] += 1
                by_key[key] = entry

    merged = sorted(by_key.values(), key=lambda x: x.get("_confidence", 0), reverse=True)
    for item in merged:
        item.pop("_confidence", None)
        item.pop("_source", None)
    logger.info(f"  [merge] {len(merged)} unique cars")
    return merged

_OLLAMA_SCRIPT_HEAD = r'''
# ── Ollama AI (same high-precision prompts as run.py) ────────────────────────
# Set USE_OLLAMA=0 to disable. AI_MODE: hybrid (default) | merge | first | off
# Requires: ollama serve  &&  ollama pull qwen2.5:7b


def _normalise_price(raw):
    p = re.sub(r"\s*/\s*", "/", re.sub(r"\s+", "", str(raw)))
    if not p.upper().startswith("CA"):
        p = "CA" + p
    return p


def _ollama_available():
    if not USE_OLLAMA or AI_MODE == "off":
        return False
    try:
        requests.get(f"{OLLAMA_BASE}/api/tags", timeout=4).raise_for_status()
        return True
    except Exception:
        return False


def _ollama_chat(messages, model=None, temperature=0.0, max_tokens=8192):
    model = model or EXTRACTOR_MODEL
    payload = {
        "model": model, "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "messages": messages,
    }
    resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def _parse_json_array(raw):
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(raw[start:end + 1])
        except Exception:
            pass
    return []


'''
_OLLAMA_SCRIPT_TAIL = r'''
def _ollama_enrich(cars, page_text, location, site_hint=""):
    """Fill Unknown car_name/car_type only — never overwrite good parser values."""
    needs = [c for c in cars
             if c.get("car_name") in (None, "", "Unknown")
             or c.get("car_type") in (None, "", "Unknown")]
    if not needs:
        return cars
    system = (
        "You are a car-rental data enrichment engine.\n"
        "ONLY fill car_name and car_type where the value is 'Unknown' or missing.\n"
        "NEVER change price_per_day, transmission, seats, bags, or location.\n"
        "NEVER invent new vehicles. Match each record to the page text by price.\n"
        "Output ONLY a JSON array with EXACTLY the same number of records, same order."
    )
    user = (
        f"Site: {site_hint or location}\n\n"
        f"Records to enrich:\n{json.dumps(needs, indent=2)}\n\n"
        f"Page text:\n{page_text[:18000]}\n\n"
        f"Return EXACTLY {len(needs)} records in the SAME ORDER."
    )
    try:
        raw = _ollama_chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        enriched = _parse_json_array(raw)
        if len(enriched) != len(needs):
            log(f"  [ollama] enrich count mismatch ({len(enriched)} vs {len(needs)})")
            return cars
        for orig, enr in zip(needs, enriched):
            if orig.get("car_name") in (None, "", "Unknown") and enr.get("car_name"):
                orig["car_name"] = enr["car_name"]
            if orig.get("car_type") in (None, "", "Unknown") and enr.get("car_type"):
                orig["car_type"] = enr["car_type"]
        log(f"  [ollama] enriched {len(enriched)} fields")
    except Exception as e:
        log(f"  [ollama] enrich failed: {e}")
    return cars


def _merge_cars_by_price(*sources):
    """One record per price; first source wins, fill missing fields from later."""
    by_price = {}
    for source in sources:
        for c in source:
            pk = re.sub(r"[^\d.]", "", str(c.get("price_per_day", ""))).strip(".")
            if not pk:
                continue
            if pk not in by_price:
                by_price[pk] = dict(c)
            else:
                base = by_price[pk]
                for field in ("seats", "bags", "transmission", "car_type", "car_name"):
                    if base.get(field) in (None, "", "Unknown") and c.get(field) not in (None, "", "Unknown"):
                        base[field] = c[field]
    return list(by_price.values())


def _finalize_cars_with_ollama(cars, page_text, site_hint):
    """Apply Ollama using the same hybrid/merge/first modes as run.py."""
    if not _ollama_available():
        if USE_OLLAMA and AI_MODE != "off":
            log("  Ollama not reachable — using parser results only (run: ollama serve)")
        return cars

    if AI_MODE == "first":
        log("  AI-first mode: reading all offers with Ollama...")
        ai_cars = _ollama_strict_extract(page_text, LOCATION, site_hint)
        parser_cars = cars
        if ai_cars and parser_cars:
            log("  Backfilling transmission/seats/bags from parser...")
            by_price = {
                re.sub(r"[^\d.]", "", str(c.get("price_per_day", ""))).strip("."): c
                for c in parser_cars
                if re.sub(r"[^\d.]", "", str(c.get("price_per_day", ""))).strip(".")
            }
            for car in ai_cars:
                pk = re.sub(r"[^\d.]", "", str(car.get("price_per_day", ""))).strip(".")
                parser = by_price.get(pk)
                if not parser:
                    continue
                for field in ("seats", "bags", "transmission", "car_type", "car_name"):
                    if car.get(field) in (None, "", "Unknown") and parser.get(field) not in (None, "", "Unknown"):
                        car[field] = parser[field]
            if len(ai_cars) < len(parser_cars):
                log(f"  Ollama found {len(ai_cars)}/{len(parser_cars)} — adding parser offers AI missed")
                ai_cars = _merge_cars_by_price(ai_cars, parser_cars)
        return ai_cars or parser_cars

    ai_cars = []
    if AI_MODE == "merge":
        log("  AI-merge mode: Ollama + parser...")
        ai_cars = _ollama_strict_extract(page_text, LOCATION, site_hint)

    if not cars:
        log("  Parser found nothing — asking Ollama to read the page...")
        ai_cars = ai_cars or _ollama_strict_extract(page_text, LOCATION, site_hint)
        return ai_cars

    if ai_cars:
        cars = _merge_cars_by_price(cars, ai_cars)
    elif any(c.get("car_name") in (None, "", "Unknown") for c in cars):
        log("  Some names missing — asking Ollama...")
        ai_cars = _ollama_strict_extract(page_text, LOCATION, site_hint)
        if ai_cars:
            cars = _merge_cars_by_price(cars, ai_cars)

    if any(c.get("car_name") in (None, "", "Unknown") for c in cars):
        cars = _ollama_enrich(cars, page_text, LOCATION, site_hint)

    return cars

'''


def _ollama_script_src() -> str:
    """Full Ollama block for generated scripts — prompts stay in sync with run.py."""
    return _OLLAMA_SCRIPT_HEAD + _ollama_script_prompt_src() + _OLLAMA_SCRIPT_TAIL


_SIXT_PARSER_SRC = r'''
# ── extraction helpers ───────────────────────────────────────────────────────

MODEL_PAREN_RE = re.compile(r"\(([^)]+)\)")
TITLE_NAME_RE = re.compile(
    r"^[A-Z][a-zA-Z\-]+(?:\s[A-Z][a-zA-Z0-9\-]+){1,4}(?:\s\(or similar\))?$"
)
FALSE_POS = {"Automatic","Manual","Standard","Select","Filter","Sort","Price",
             "Day","Per","Choose","Search","Rated","Pick","View","All Cars",
             "Recommended","Featured","Pay Later","Unlimited","Kilometers"}
_SIXT_TYPE_SET = {t.upper() for t in CAR_TYPES}
_SIXT_MODEL_LINE_RE = re.compile(r"^[A-Z0-9][A-Z0-9 \-]{2,45}$")


def _normalise_price(raw):
    p = re.sub(r"\s*/\s*", "/", re.sub(r"\s+", "", raw))
    if not p.upper().startswith("CA"):
        p = "CA" + p
    return p


def _parse_seats_bags(lines):
    seats = bags = None
    stop = len(lines)
    for j, w in enumerate(lines):
        if w.lower().startswith("unlimited") or PRICE_RE.search(w):
            stop = j
            break
    nums = []
    for w in lines[:stop]:
        sm = re.match(r"^(\d+)\s*(?:\(\d+\+\d+\))?\s*$", w)
        if sm:
            n = int(sm.group(1))
            if 1 <= n <= 9:
                nums.append(n)
    if len(nums) >= 2:
        seats, bags = nums[-2], nums[-1]
    elif len(nums) == 1:
        seats = nums[0]
    return seats, bags


def _sixt_type_rank(type_up):
    if type_up in ("SUV", "MINIVAN", "VAN", "PICKUP", "TRUCK", "CROSSOVER"):
        return 3
    if type_up in ("PREMIUM", "LUXURY", "FULLSIZE", "FULL-SIZE", "INTERMEDIATE", "COMPACT"):
        return 2
    if type_up == "STANDARD":
        return 1
    return 2


def _resolve_sixt_car_header(window):
    car_name, car_type = "Unknown", "Unknown"
    header_slice = window
    for j, w in enumerate(window):
        hm = SIXT_CLASS_LINE_RE.match(w)
        if hm:
            base_type = hm.group(1).replace("-", "").title()
            if base_type.lower() == "fullsize":
                base_type = "Fullsize"
            car_type = f"{base_type} Elite" if hm.group(2) else base_type
            car_name = hm.group(3).strip()
            return car_name, car_type, window[j:]
    best_split = None
    for j, w in enumerate(window):
        w_st = w.strip()
        w_up = w_st.upper()
        if w_up not in _SIXT_TYPE_SET:
            continue
        if w_up == "STANDARD" and j > 0 and window[j - 1].strip().lower() in (
            "select", "filter", "sort", "recommended",
        ):
            continue
        type_label = "Fullsize" if w_up.replace("-", "") == "FULLSIZE" else w_st.title()
        for k in range(j + 1, min(j + 5, len(window))):
            nxt = window[k].strip()
            if not nxt or nxt.lower() in ("automatic", "manual"):
                continue
            if PRICE_RE.search(nxt):
                break
            if _SIXT_MODEL_LINE_RE.match(nxt) and len(nxt.split()) >= 2:
                if nxt.upper() not in _SIXT_TYPE_SET and nxt not in FALSE_POS:
                    rank = _sixt_type_rank(w_up)
                    if best_split is None or (k, rank) > (best_split[0], best_split[1]):
                        best_split = (k, rank, nxt, type_label, window[j:])
    if best_split:
        _, _, car_name, car_type, header_slice = best_split
        return car_name, car_type, header_slice
    for w in reversed(window):
        w_st = w.strip()
        if w_st in FALSE_POS or w_st.lower() in ("automatic", "manual", "unlimited"):
            continue
        if PRICE_RE.search(w_st):
            continue
        if _SIXT_MODEL_LINE_RE.match(w_st) and len(w_st.split()) >= 2:
            if w_st.upper() not in _SIXT_TYPE_SET:
                car_name = w_st
                break
        pm = MODEL_PAREN_RE.search(w_st)
        if pm and car_name == "Unknown":
            candidate = pm.group(1).strip()
            if candidate.lower() not in {"or similar", "awd", "4wd", "4x4"}:
                car_name = candidate
        if car_name == "Unknown" and TITLE_NAME_RE.match(w_st) and len(w_st.split()) >= 2:
            car_name = w_st
    if car_type == "Unknown":
        for w in reversed(window):
            tm = TYPE_RE.search(w)
            if tm:
                car_type = tm.group(1).title()
                break
            w_up = w.strip().upper()
            if w_up in _SIXT_TYPE_SET:
                car_type = "Fullsize" if w_up.replace("-", "") == "FULLSIZE" else w.strip().title()
                break
    return car_name, car_type, header_slice


def _extract_sixt_cards(page_text):
    lines = [l.strip() for l in page_text.split("\n") if l.strip()]
    cars = []
    last_price_idx = -1
    for i, line in enumerate(lines):
        price_m = PRICE_RE.search(line)
        if not price_m:
            continue
        raw_price = _normalise_price(price_m.group(1))
        num_m = NUM_RE.search(raw_price)
        if not num_m:
            continue
        try:
            val = float(num_m.group().replace(",", ""))
        except ValueError:
            continue
        if not (10.0 <= val <= 5000.0):
            continue
        window = lines[last_price_idx + 1:i]
        last_price_idx = i
        car_name, car_type, header_slice = _resolve_sixt_car_header(window)
        transmission = "Automatic"
        seats, bags = _parse_seats_bags(header_slice)
        for w in reversed(window):
            if re.search(r"\b(automatic|manual)\b", w, re.I):
                tr = TRANS_RE.search(w)
                if tr:
                    transmission = tr.group(1).title()
                break
        if car_name == "Unknown":
            continue
        cars.append({
            "car_name": car_name, "car_type": car_type,
            "price_per_day": raw_price, "transmission": transmission,
            "seats": seats, "bags": bags, "location": LOCATION,
            "_source": "sixt_parser",
        })
    seen, unique = set(), []
    for c in cars:
        key = (c["car_name"].lower(), c["price_per_day"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(c)
    log(f"  [parser] {len(unique)} cars")
    return unique


def _extract_dom_sixt(page):
    try:
        chunks = page.evaluate(r"""
            () => {
                const out = [], seen = new Set();
                const nodes = document.querySelectorAll(
                    '[class*="offer"],[class*="vehicle"],[class*="car"],[data-testid*="offer"],article'
                );
                for (const el of nodes) {
                    const t = (el.innerText || '').trim();
                    if (!t || t.length < 40 || t.length > 2500) continue;
                    if (!/CA\$[\d,.]+\s*\/\s*day/i.test(t)) continue;
                    const key = t.slice(0, 120);
                    if (seen.has(key)) continue;
                    seen.add(key);
                    out.push(t);
                }
                return out;
            }
        """)
        merged = []
        for chunk in chunks or []:
            merged.extend(_extract_sixt_cards(chunk))
        log(f"  [dom] {len(merged)} cars")
        return merged
    except Exception as e:
        log(f"  [dom] failed: {e}")
        return []


def _sanity_filter(cars):
    good = []
    for c in cars:
        m = NUM_RE.search(str(c.get("price_per_day", "")))
        if not m:
            continue
        try:
            val = float(m.group().replace(",", ""))
        except ValueError:
            continue
        if 10.0 <= val <= 1500.0:
            good.append(c)
    return good


def _norm_price_key(price):
    return re.sub(r"[^\d.]", "", str(price or "")).strip(".")


def _merge_sixt_by_price(*sources):
    """One car per price; first source wins (pass DOM before page parser)."""
    by_price = {}
    for source in sources:
        for c in source:
            pk = _norm_price_key(c.get("price_per_day", ""))
            if not pk:
                continue
            if pk not in by_price:
                by_price[pk] = dict(c)
            else:
                base = by_price[pk]
                for field in ("seats", "bags", "transmission", "car_type", "car_name"):
                    if base.get(field) in (None, "", "Unknown") and c.get(field) not in (None, "", "Unknown"):
                        base[field] = c[field]
    return list(by_price.values())

'''

def _sixt_metadata_helpers_src() -> str:
    """Python source embedded in generated Sixt scripts — mirrors run.py metadata."""
    js = _SIXT_METADATA_JS.strip()
    return f'''# ── rental date/time metadata (matches run.py) ─────────────────────────────

def _normalize_time_str(time_str):
    raw = str(time_str or "").strip()
    if not raw:
        return ""
    raw = raw.upper().replace(".", ":")
    m = re.match(r"(\\d{{1,2}}):(\\d{{2}})\\s*(AM|PM)?", raw, re.I)
    if not m:
        return raw[:5]
    hour, minute, ampm = int(m.group(1)), m.group(2), (m.group(3) or "").upper()
    if ampm == "PM" and hour < 12:
        hour += 12
    elif ampm == "AM" and hour == 12:
        hour = 0
    return f"{{hour:02d}}:{{minute}}"


def _to_iso_date(date_str):
    raw = str(date_str or "").strip()
    if not raw:
        raise ValueError("empty date")
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"unrecognized date: {{date_str!r}}")


def _parse_rental_field(val):
    raw = str(val or "").strip()
    if not raw:
        return None, None
    if "T" in raw or re.fullmatch(r"\\d{{4}}-\\d{{2}}-\\d{{2}}", raw):
        if "T" in raw:
            date_part, time_part = raw.split("T", 1)
            return date_part[:10], _normalize_time_str(time_part)
        return raw, None
    tm = re.search(r"(\\d{{1,2}}:\\d{{2}}\\s*(?:AM|PM)?)", raw, re.I)
    if tm:
        date_part = raw[: tm.start()].strip(" ,|-")
        try:
            return _to_iso_date(date_part), _normalize_time_str(tm.group(1))
        except ValueError:
            return None, _normalize_time_str(tm.group(1))
    try:
        return _to_iso_date(raw), None
    except ValueError:
        return None, _normalize_time_str(raw)


_SIXT_METADATA_JS = r"""
{js}
"""


def _normalize_sixt_metadata(raw):
    out = {{}}
    field_map = [
        ("pickup_date_display", "pickup_date", "pickup_time"),
        ("pickupDate", "pickup_date", "pickup_time"),
        ("pickup_date", "pickup_date", "pickup_time"),
        ("url_pickup", "pickup_date", "pickup_time"),
        ("pickup_storage", "pickup_date", "pickup_time"),
        ("return_date_display", "return_date", "return_time"),
        ("returnDate", "return_date", "return_time"),
        ("return_date", "return_date", "return_time"),
        ("url_return", "return_date", "return_time"),
        ("return_storage", "return_date", "return_time"),
        ("pickup_time_display", None, "pickup_time"),
        ("return_time_display", None, "return_time"),
        ("pickup_time_storage", None, "pickup_time"),
        ("return_time_storage", None, "return_time"),
    ]
    for src_key, date_key, time_key in field_map:
        val = (raw or {{}}).get(src_key)
        if not val:
            continue
        iso, tm = _parse_rental_field(val)
        if iso and date_key and date_key not in out:
            out[date_key] = iso
        if tm and time_key and time_key not in out:
            out[time_key] = tm
    return out


def _read_sixt_rental_metadata(page):
    try:
        raw = page.evaluate(_SIXT_METADATA_JS)
    except Exception:
        raw = {{}}
    return _normalize_sixt_metadata(raw or {{}})


def _accumulate_rental_metadata(base, extra):
    out = {{k: v for k, v in base.items() if v}}
    for key, val in extra.items():
        if val:
            out[key] = val
    return out


def _finalize_rental_metadata(page_meta, user_meta=None, site="sixt"):
    user_meta = user_meta or {{}}
    out = {{k: v for k, v in page_meta.items() if v}}
    for key, val in user_meta.items():
        if val:
            out[key] = val
    if not out.get("pickup_date") or not out.get("return_date"):
        now = now_mst()
        if site == "sixt":
            out.setdefault("pickup_date", (now + timedelta(days=2)).strftime("%Y-%m-%d"))
            out.setdefault("return_date", (now + timedelta(days=5)).strftime("%Y-%m-%d"))
        else:
            out.setdefault("pickup_date", (now + timedelta(days=1)).strftime("%Y-%m-%d"))
            out.setdefault("return_date", (now + timedelta(days=2)).strftime("%Y-%m-%d"))
    return out


def _stamp_rental_dates(cars, meta):
    for car in cars:
        for key in ("pickup_date", "return_date", "pickup_time", "return_time"):
            if meta.get(key):
                car[key] = meta[key]
    return cars

'''


_SIXT_SCRIPT_TAIL = r'''

# ── Sixt search flow (matches run_sixt — preloaded dates, no calendar) ───────

_LOC_SKIP = {"airport", "international", "intl", "int", "city", "the"}

_SIXT_SELECT_BEST_JS = r"""
({query, requirePrimary}) => {
  const q = (query || '').toLowerCase();
  const words = q.split(/\s+/).filter(Boolean);
  const skip = new Set(['airport', 'international', 'intl', 'int', 'city', 'the']);
  const primary = words.find(w => !skip.has(w)) || words[0] || '';
  const sels = '[data-testid*="suggestion"],[class*="suggestion"],[role="option"],'
             + '[class*="result"] li,[class*="autocomplete"] li,ul[role="listbox"] li,'
             + '[id*="downshift"] li,[id*="react-select"] [role="option"],li';
  const visible = el => {
    const r = el.getBoundingClientRect();
    const s = window.getComputedStyle(el);
    return r.width > 2 && r.height > 2 && s.display !== 'none'
        && s.visibility !== 'hidden' && s.opacity !== '0';
  };
  let items = Array.from(document.querySelectorAll(sels))
    .filter(visible)
    .filter(el => (el.innerText || '').trim().length > 2);
  if (!items.length) return {status: 'no_items'};
  const set = new Set(items);
  const leaves = items.filter(el => {
    for (const other of set) {
      if (other !== el && el.contains(other)) return false;
    }
    return true;
  });
  const pool = leaves.length ? leaves : items;
  let best = null, bestScore = -Infinity;
  for (const el of pool) {
    const raw = (el.innerText || '').trim();
    const t = raw.toLowerCase();
    let score = 0;
    for (const w of words) if (t.includes(w)) score += 1;
    if (primary && t.includes(primary)) score += 5; else if (primary) score -= 10;
    if (/\bairport\b/.test(t)) score += 1.5;
    if (el.getAttribute('role') === 'option') score += 2;
    if ((el.getAttribute('data-testid') || '').toLowerCase().includes('suggestion')) score += 2;
    score -= raw.length / 50;
    if (score > bestScore) { bestScore = score; best = el; }
  }
  if (!best) return {status: 'no_match'};
  if (requirePrimary && primary && !(best.innerText || '').toLowerCase().includes(primary)) {
    return {status: 'no_primary'};
  }
  best.scrollIntoView({block: 'center', inline: 'center'});
  const r = best.getBoundingClientRect();
  const opts = {bubbles: true, cancelable: true, view: window,
                clientX: r.left + r.width / 2, clientY: r.top + r.height / 2};
  for (const type of ['pointerover', 'pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click']) {
    best.dispatchEvent(new MouseEvent(type, opts));
  }
  return {status: 'clicked',
          text: (best.innerText || '').trim().replace(/\s+/g, ' ').slice(0, 100)};
}
"""


def _primary_token(text):
    words = [w for w in re.split(r"\s+", (text or "").lower()) if w]
    return next((w for w in words if w not in _LOC_SKIP), words[0] if words else "")


def _wait_for_sixt_search_form(page, timeout_ms=12000):
    try:
        page.wait_for_function("""
            () => {
              const read = (tid) => {
                const root = document.querySelector(`[data-testid="${tid}"]`);
                if (!root) return '';
                for (const btn of root.querySelectorAll('button')) {
                  const t = (btn.getAttribute('aria-label') || btn.innerText || '').trim();
                  if (t && !/^(pickup|return)\\s+(date|time)$/i.test(t)) return t;
                }
                return '';
              };
              return read('rent-search-form-pickup-date-input')
                  || read('rent-search-form-return-date-input')
                  || localStorage.getItem('rent-search_historyPickup');
            }
        """, timeout=timeout_ms)
    except Exception:
        pass


def _location_error_visible(page):
    try:
        return "please select a pickup location" in page.inner_text("body", timeout=1500).lower()
    except Exception:
        return False


def _commit_location(page, location):
    box = None
    for sel in [
        'input[placeholder*="Airport" i]', 'input[placeholder*="city" i]',
        'input[placeholder*="location" i]', 'input[placeholder*="pickup" i]',
        'input[type="search"]',
    ]:
        try:
            loc = page.locator(sel)
            if loc.count() > 0 and loc.first.is_visible(timeout=3000):
                box = loc.first
                break
        except Exception:
            continue
    if box is None:
        try:
            cand = page.locator("input:visible").first
            if cand.count() > 0 and cand.is_visible(timeout=2500):
                box = cand
        except Exception:
            box = None
    if box is None:
        log("  Could not find the location search box.")
        return False

    primary = _primary_token(location)

    def committed():
        try:
            body = page.inner_text("body", timeout=1200).lower()
        except Exception:
            body = ""
        return "please select a pickup location" not in body

    def select_best():
        for _ in range(16):
            try:
                res = page.evaluate(_SIXT_SELECT_BEST_JS,
                                    {"query": location, "requirePrimary": True})
            except Exception:
                res = {"status": "error"}
            if res and res.get("status") == "clicked":
                return res.get("text")
            page.wait_for_timeout(250)
        try:
            res = page.evaluate(_SIXT_SELECT_BEST_JS,
                                {"query": location, "requirePrimary": False})
            if res and res.get("status") == "clicked":
                return res.get("text")
        except Exception:
            pass
        return None

    for attempt in range(1, 4):
        try:
            box.click()
            box.fill("")
            page.wait_for_timeout(300)
            try:
                box.press_sequentially(location, delay=70)
            except Exception:
                box.type(location, delay=70)
        except Exception:
            pass
        page.wait_for_timeout(800)
        clicked = select_best()
        page.wait_for_timeout(1000)
        good_city = (not primary) or (clicked and primary in clicked.lower())
        if clicked and committed() and good_city:
            log(f"  Location accepted: {clicked}")
            return True
        page.keyboard.press("ArrowDown")
        page.wait_for_timeout(300)
        page.keyboard.press("Enter")
        page.wait_for_timeout(1200)
        if committed() and ((not primary) or attempt >= 2):
            log(f"  Location accepted: {clicked or location}")
            return True
        log(f"  Location not committed yet — retrying ({attempt}/3)")
    return committed()


def _click_show_cars(page):
    for sel in [
        "button:has-text('Show cars')", "button:has-text('SHOW CARS')",
        "button:has-text('Show Cars')", "[data-testid='show-cars-button']",
        "button:has-text('Show stations')", "button:has-text('SHOW STATIONS')",
        "button:has-text('Show Stations')", "[data-testid*='show-stations']",
        "[data-testid*='show-cars']", "form button[type='submit']",
        "button[type='submit']",
    ]:
        try:
            btn = page.locator(sel)
            if btn.count() > 0 and btn.first.is_visible(timeout=1500):
                btn.first.scroll_into_view_if_needed()
                page.wait_for_timeout(200)
                btn.first.click()
                log(f"  Show cars -> {sel}")
                return True
        except Exception:
            continue
    try:
        result = page.evaluate("""
            () => {
                for (const btn of Array.from(document.querySelectorAll('button'))) {
                    const txt = btn.innerText.trim().toLowerCase();
                    if (txt.includes('show') || txt.includes('car') || txt.includes('search')) {
                        const bg = window.getComputedStyle(btn).backgroundColor;
                        if (bg && bg !== 'transparent' && bg !== 'rgba(0, 0, 0, 0)') {
                            btn.click();
                            return btn.innerText.trim();
                        }
                    }
                }
                return null;
            }
        """)
        if result:
            log(f"  Show cars JS -> '{result}'")
            return True
    except Exception:
        pass
    log("  WARNING: Show cars button not found")
    return False


def _is_branch_page(page):
    try:
        if "nearbybranches" in page.url.lower():
            return True
        body = page.inner_text("body")[:2500].lower()
        return "select a pickup branch" in body or "pickup branch" in body
    except Exception:
        return False


def _first_branch_name(page):
    try:
        return page.evaluate(r"""
            () => {
                const skip = /fully booked/i;
                const lines = document.body.innerText.split('\n').map(l=>l.trim()).filter(Boolean);
                let pastHeader = false;
                for (let i = 0; i < lines.length; i++) {
                    const line = lines[i];
                    if (/select a pickup branch/i.test(line)) { pastHeader=true; continue; }
                    if (!pastHeader) continue;
                    if (skip.test(line)) continue;
                    if (/^(starting at|ca\$|keyboard|map data|help|log in)/i.test(line)) continue;
                    if (line.length < 5 || line.length > 55) continue;
                    const nxt = (lines[i+1]||'').toLowerCase();
                    const nxt2= (lines[i+2]||'').toLowerCase();
                    if (nxt.includes('starting at')||nxt2.includes('starting at')) return line;
                }
                return null;
            }
        """)
    except Exception:
        return None


def _select_branch(page):
    page.wait_for_timeout(1500)
    branch_name = _first_branch_name(page)
    if branch_name:
        log(f"  First branch: {branch_name}")
        try:
            loc = page.get_by_text(branch_name, exact=True)
            if loc.count() > 0:
                loc.first.scroll_into_view_if_needed()
                loc.first.click(timeout=5000)
                page.wait_for_timeout(2000)
                return True
        except Exception:
            pass
    try:
        picked = page.evaluate(r"""
            () => {
                const skip = /fully booked/i;
                const rows = [];
                for (const el of document.querySelectorAll('div,li,article,button,[role="button"]')) {
                    const t = (el.innerText||'').trim();
                    if (!t||t.length>400||t.length<15) continue;
                    if (skip.test(t)) continue;
                    if (!/starting at/i.test(t)&&!/CA\$[\d,.]+\s*\/\s*day/i.test(t)) continue;
                    const r = el.getBoundingClientRect();
                    if (r.left>480||r.width<100||r.top<80) continue;
                    rows.push({el, top:r.top, t:t.split('\n')[0].slice(0,50)});
                }
                rows.sort((a,b)=>a.top-b.top);
                if (rows.length) { rows[0].el.click(); return rows[0].t; }
                return null;
            }
        """)
        if picked:
            log(f"  Branch row: {picked}")
            page.wait_for_timeout(2000)
            return True
    except Exception:
        pass
    return False


def _click_show_offers(page):
    try:
        page.wait_for_function(r"""
            () => Array.from(document.querySelectorAll('button, a, [role="button"]'))
                .some(el => /show\s+offers?/i.test((el.innerText || '').trim()))
        """, timeout=8000)
    except Exception:
        pass
    page.wait_for_timeout(500)
    for sel in ["button:has-text('Show offers')", "button:has-text('SHOW OFFERS')",
                "a:has-text('Show offers')", "[data-testid*='show-offers']",
                "button:has-text('Show cars')"]:
        try:
            btn = page.locator(sel)
            for i in range(min(btn.count(), 3)):
                item = btn.nth(i)
                if item.is_visible(timeout=1500):
                    item.scroll_into_view_if_needed()
                    item.click(timeout=5000)
                    log(f"  Show offers -> {sel}")
                    page.wait_for_timeout(5000)
                    return True
        except Exception:
            continue
    return False


def _branch_to_offers(page):
    if not _is_branch_page(page):
        return False
    log("Branch selection page detected")
    for attempt in range(1, 4):
        log(f"  Attempt {attempt}/3...")
        _select_branch(page)
        page.wait_for_timeout(1500)
        if _click_show_offers(page):
            try:
                page.wait_for_function(
                    r"""() => /which car do you want|CA\$[\d,.]+\/\s*day/i.test(document.body.innerText)""",
                    timeout=15000,
                )
                log("  Car offers page loaded")
            except Exception:
                log("  Car offers wait timed out — continuing")
            page.wait_for_timeout(3000)
            return True
        branch_name = _first_branch_name(page)
        if branch_name:
            try:
                page.get_by_text(branch_name, exact=True).first.click(timeout=3000)
                page.wait_for_timeout(1500)
            except Exception:
                pass
    log("  All Show offers attempts failed")
    return False


def scrape():
    cars = []
    rental_meta = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=HEADLESS,
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = browser.new_page(user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ))
        try:
            log("Navigating to https://www.sixt.ca ...")
            page.goto("https://www.sixt.ca", wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(3000)

            log("Dismissing cookie popup...")
            _dismiss_popups(page)

            log("Waiting for Sixt search form...")
            _wait_for_sixt_search_form(page)

            log(f"Searching for location: {LOCATION}")
            if not _commit_location(page, LOCATION):
                log("Location not accepted — aborting")
                page.screenshot(path=os.path.join(OUTPUT_DIR, "location_error.png"))
                return []

            rental_meta = _accumulate_rental_metadata(
                rental_meta, _read_sixt_rental_metadata(page))

            log("Using Sixt's preloaded pickup/return dates (no calendar override)")

            log("Clicking Show cars / Show stations...")
            if not _click_show_cars(page):
                log("Could not open car list — aborting")
                page.screenshot(path=os.path.join(OUTPUT_DIR, "show_cars_error.png"))
                return []

            if _location_error_visible(page):
                log("Sixt rejected location — retrying dropdown selection")
                if not _commit_location(page, LOCATION):
                    return []
                if not _click_show_cars(page):
                    return []

            try:
                page.wait_for_load_state("domcontentloaded", timeout=20000)
            except Exception:
                pass
            page.wait_for_timeout(4000)

            if _is_branch_page(page) or "station" in page.inner_text("body")[:1500].lower():
                log("Choosing pickup branch...")
                _branch_to_offers(page)

            log("Scrolling to load all cars...")
            last_h = page.evaluate("document.body.scrollHeight")
            for i in range(15):
                page.evaluate("window.scrollBy(0, 800)")
                page.wait_for_timeout(800)
                new_h = page.evaluate("document.body.scrollHeight")
                if i > 5 and new_h == last_h:
                    break
                last_h = new_h
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(2000)
            page.screenshot(path=os.path.join(OUTPUT_DIR, "results.png"))

            page_text = page.inner_text("body")
            with open(os.path.join(OUTPUT_DIR, "page_content.txt"), "w") as f:
                f.write(page_text)
            log(f"Page text: {len(page_text)} chars")

            price_hits = len(re.findall(r"CA\$[\d,]+\.?\d*/day", page_text, re.IGNORECASE))
            log(f"{price_hits} CA$/day patterns found")

            log("Running Sixt parser...")
            page_cars = _extract_sixt_cards(page_text)
            log("Running DOM extractor...")
            dom_cars = _extract_dom_sixt(page)

            cars = _merge_sixt_by_price(dom_cars, page_cars)
            log(f"Merged: {len(cars)} unique cars (by price)")

            rental_meta = _accumulate_rental_metadata(
                rental_meta, _read_sixt_rental_metadata(page))
            rental_meta = _finalize_rental_metadata(rental_meta, site="sixt")
            if rental_meta.get("pickup_date"):
                log(f"Rental dates: {rental_meta.get('pickup_date')} {rental_meta.get('pickup_time', '')} -> "
                    f"{rental_meta.get('return_date')} {rental_meta.get('return_time', '')}".rstrip())
            cars = _stamp_rental_dates(cars, rental_meta)

            cars = _sanity_filter(cars)
            cars = _finalize_cars_with_ollama(cars, page_text, SITE_HINT)
            cars = _sanity_filter(cars)
            log(f"Final: {len(cars)} valid cars")

        except Exception as e:
            log(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            try:
                page.screenshot(path=os.path.join(OUTPUT_DIR, "error.png"))
            except Exception:
                pass
        finally:
            browser.close()
    return cars


def save(cars):
    if not cars:
        log("No cars to save.")
        return
    stamp = now_mst().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(cars)
    extra_cols = [c for c in df.columns if c not in CSV_COLUMNS]
    df = df[[c for c in CSV_COLUMNS if c in df.columns] + extra_cols]
    csv_path  = os.path.join(OUTPUT_DIR, f"{stamp}.csv")
    json_path = os.path.join(OUTPUT_DIR, f"{stamp}.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(cars, f, indent=2, ensure_ascii=False)
    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  {len(cars)} vehicles  |  {now_mst().strftime('%Y-%m-%d %H:%M:%S MST')}")
    print(f"  Location : {LOCATION}")
    print(f"  CSV      : {csv_path}")
    print(f"  JSON     : {json_path}")
    print(f"{sep}")
    print("\nFirst 5 results:")
    for i, c in enumerate(cars[:5], 1):
        print(f"  {i}. {c['car_name']} ({c['car_type']}) — {c['price_per_day']}")


if __name__ == "__main__":
    log(f"Starting Sixt scraper for: {LOCATION}")
    log("Dates: using Sixt's preloaded search form values")
    cars = scrape()
    save(cars)
'''


def _save_script(site, location, folder, stamp):
    import textwrap  # Added to ensure the Sixt block formatting works
    
    safe_loc      = location.replace('"', '\\"')
    generated_at  = now_mst().strftime("%Y-%m-%d %H:%M:%S MST")
    pickup_date   = (now_mst() + timedelta(days=1)).strftime("%m/%d/%Y")
    return_date   = (now_mst() + timedelta(days=3)).strftime("%m/%d/%Y")
    pickup_iso    = (now_mst() + timedelta(days=1)).strftime("%Y-%m-%d")
    return_iso    = (now_mst() + timedelta(days=3)).strftime("%Y-%m-%d")

    site_hint = "Sixt.ca" if "sixt" in site.lower() else "Enterprise.com"

    # ── shared boilerplate header ────────────────────────────────────────────
    HEADER = f'''\
#!/usr/bin/env python3
"""
{'='*64}
  Site     : {site}
  Location : {location}
  Generated: {generated_at}
  Pickup   : {pickup_iso}
  Return   : {return_iso}
{'='*64}
Run directly:
    python3 {stamp}.py

Optional AI (Ollama — same prompts as run.py):
    ollama serve && ollama pull qwen2.5:7b
    python3 {stamp}.py
    USE_OLLAMA=0 python3 {stamp}.py          # parsers only
    AI_MODE=merge python3 {stamp}.py         # parser + Ollama merge
    AI_MODE=first python3 {stamp}.py         # Ollama drives extraction
"""

import os, re, json, logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from playwright.sync_api import sync_playwright, Page

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

LOCATION        = "{safe_loc}"
SITE_HINT       = "{site_hint}"
PICKUP_DATE     = "{pickup_date}"
RETURN_DATE     = "{return_date}"
HEADLESS        = False
USE_OLLAMA      = os.environ.get("USE_OLLAMA", "1") not in ("0", "", "false", "False")
AI_MODE         = os.environ.get("AI_MODE", "hybrid").lower()
OLLAMA_BASE     = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
EXTRACTOR_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
MST             = ZoneInfo("America/Edmonton")

def now_mst():
    return datetime.now(MST)

def mst_log_prefix():
    return now_mst().strftime("%H:%M:%S")

def log(msg):
    print(f"[{{mst_log_prefix()}}] {{msg}}")

OUTPUT_DIR = os.path.join(
    "scraper_outputs", "{site}", re.sub(r'[^a-zA-Z0-9]', '_', LOCATION)[:28]
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_COLUMNS = [
    "pickup_date", "return_date", "pickup_time", "return_time",
    "car_name", "car_type", "price_per_day", "transmission", "seats", "bags", "location",
]

PRICE_RE = re.compile(r"((?:CA)?\\s*\\$\\s*[\\d,]+\\.?\\d*\\s*/\\s*day)", re.IGNORECASE)
NUM_RE   = re.compile(r"[\\d,]+\\.?\\d*")

ENTERPRISE_CARD_RE = re.compile(
    r"(?P<category>[A-Za-z0-9][A-Za-z0-9 \\-]+?)\\s*\\n+"
    r"(?P<model>[^\\n]+? or similar)\\s*\\n+Automatic\\s*\\n+"
    r"(?P<seats>\\d+)\\s+People\\s*\\n+"
    r"(?P<bags>\\d+)\\s+Bags\\s*\\n+"
    r"(?:[^\\n]*\\n){{0,20}}?"
    r"(?P<per_day>\\$[\\d,]+\\.?\\d*)\\s*\\n+Per Day",
    re.IGNORECASE | re.MULTILINE,
)
FALSE_POS = {{"Automatic","Manual","Standard","Select","Filter","Sort","Price",
             "Day","Per","Choose","Search","Rated","Pick","View","All Cars",
             "Recommended","Featured","Pay Later","Unlimited","Kilometers"}}

'''

    # ── Enterprise body ──────────────────────────────────────────────────────
    if "enterprise" in site.lower():
        body = HEADER + _ollama_script_src() + '''\

# ── helpers ─────────────────────────────────────────────────────────────────

def _usd_to_ca_per_day(usd_str):
    # Fixed escape sequence warning below by using double backslashes
    m = re.search(r"([\\d,]+\\.?\\d*)", usd_str.replace("$", ""))
    if not m:
        return ""
    val = m.group(1).replace(",", "")
    return f"CA${val}/day"

def _normalize_enterprise_text(page_text):
    lines = page_text.split("\\n")
    out, i = [], 0
    while i < len(lines):
        s = lines[i].strip()
        if s == "$" and i + 2 < len(lines):
            whole, frac = lines[i+1].strip(), lines[i+2].strip()
            if re.match(r"^\\d+$", whole) and re.match(r"^\\..+$", frac):
                out.append(f"${whole}{frac}")
                i += 3
                continue
        out.append(lines[i])
        i += 1
    return "\\n".join(out)

def _truncate_enterprise_listing(page_text):
    low = page_text.lower()
    cut = len(page_text)
    for marker in ("hide alternative", "explore alternative"):
        idx = low.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return page_text[:cut]

def _extract_enterprise_cards(page_text):
    text = _normalize_enterprise_text(_truncate_enterprise_listing(page_text))
    cars = []
    for m in ENTERPRISE_CARD_RE.finditer(text):
        category = m.group("category").strip()
        model    = m.group("model").strip()
        if category.lower() in {p.lower() for p in FALSE_POS}:
            continue
        per_day = _usd_to_ca_per_day(m.group("per_day"))
        if not per_day:
            continue
        cars.append({
            "car_name"     : model,
            "car_type"     : category,
            "price_per_day": per_day,
            "transmission" : "Automatic",
            "seats"        : int(m.group("seats")),
            "bags"         : int(m.group("bags")),
            "location"     : LOCATION,
        })
    seen, unique = set(), []
    for c in cars:
        key = (c["car_name"].lower(), c["price_per_day"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(c)
    log(f"  [parser] {len(unique)} cars")
    return unique

def _extract_dom_enterprise(page):
    try:
        chunks = page.evaluate("""
            () => {
                const out = [], seen = new Set();
                for (const el of document.querySelectorAll(
                    '[class*="vehicle"],[class*="Vehicle"],article,[data-testid*="vehicle"]'
                )) {
                    const t = (el.innerText || '').trim();
                    if (!t || t.length < 50 || t.length > 2000) continue;
                    if (!/or similar/i.test(t)) continue;
                    if (!/People/i.test(t) || !/Per Day/i.test(t)) continue;
                    const key = t.slice(0, 100);
                    if (seen.has(key)) continue;
                    seen.add(key);
                    out.push(t);
                }
                return out;
            }
        """)
        merged = []
        for chunk in chunks or []:
            merged.extend(_extract_enterprise_cards(chunk))
        log(f"  [dom] {len(merged)} cars")
        return merged
    except Exception as e:
        log(f"  [dom] failed: {e}")
        return []

def _sanity_filter(cars):
    good = []
    for c in cars:
        m = NUM_RE.search(str(c.get("price_per_day", "")))
        if not m:
            continue
        try:
            val = float(m.group().replace(",", ""))
        except ValueError:
            continue
        if 10.0 <= val <= 1500.0:
            good.append(c)
    return good

def _dismiss_popups(page):
    SELS = [
        "button:has-text('I AGREE')", "button:has-text('Accept all')",
        "button:has-text('Accept')", "button:has-text('Close')",
        "#onetrust-accept-btn-handler", "button[aria-label='Close']",
    ]
    for sel in SELS:
        try:
            loc = page.locator(sel)
            if loc.count() > 0 and loc.first.is_visible(timeout=1500):
                loc.first.click(timeout=3000)
                log(f"  Popup dismissed: {sel}")
                page.wait_for_timeout(1200)
                return
        except Exception:
            continue

def _dismiss_enterprise_promos(page):
    log("  Waiting 5s for promo banners...")
    page.wait_for_timeout(5000)
    PROMO_SELS = [
        "button[aria-label='Close']", "button[aria-label='close']",
        "[class*='modal'] button[aria-label*='lose' i]",
        "[class*='overlay'] button[aria-label*='lose' i]",
        "[class*='dialog'] [class*='close']",
    ]
    for sel in PROMO_SELS:
        try:
            loc = page.locator(sel)
            if loc.count() > 0 and loc.first.is_visible(timeout=1500):
                loc.first.click(timeout=3000)
                log(f"  Promo closed: {sel}")
                page.wait_for_timeout(1000)
                return
        except Exception:
            continue
    try:
        page.keyboard.press("Escape")
        page.wait_for_timeout(800)
    except Exception:
        pass
    try:
        page.evaluate("""
            () => {
                for (const sel of ['[role="dialog"]','[aria-modal="true"]',
                                   '[class*="modal"]','[class*="overlay"]']) {
                    for (const el of document.querySelectorAll(sel)) {
                        const r = el.getBoundingClientRect();
                        if (r.width > 300 && r.height > 200)
                            el.style.display = 'none';
                    }
                }
            }
        """)
    except Exception:
        pass
    page.wait_for_timeout(1500)

def _fill_date(page, locator, value):
    try:
        locator.scroll_into_view_if_needed()
        locator.click()
        page.wait_for_timeout(300)
        locator.press("Control+a")
        locator.type(value, delay=80)
        page.wait_for_timeout(200)
        locator.press("Tab")
        return True
    except Exception:
        return False

# ── main scrape ──────────────────────────────────────────────────────────────

def scrape():
    cars = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=HEADLESS,
            args=["--disable-blink-features=AutomationControlled"]
        )
        page = browser.new_page(user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ))
        try:
            # 1. Load page
            log("Navigating to Enterprise locations...")
            page.goto(
                "https://www.enterprise.com/en/car-rental/locations.html",
                wait_until="domcontentloaded", timeout=30000
            )
            page.wait_for_timeout(1500)

            # 2. Dismiss popups + promos
            log("Dismissing popups...")
            _dismiss_popups(page)
            log("Dismissing promo overlays...")
            _dismiss_enterprise_promos(page)

            # 3. Type location
            log(f"Typing location: {LOCATION}")
            typed = False
            for sel in ["input#pickupLocationTextBox", "input#geoLocation",
                        "input[id*='location' i]", "input[placeholder*='city' i]",
                        "input[placeholder*='airport' i]", "input[type='text']:visible"]:
                try:
                    inp = page.locator(sel)
                    if inp.count() > 0 and inp.first.is_visible(timeout=3000):
                        inp.first.click()
                        inp.first.fill("")
                        page.wait_for_timeout(200)
                        inp.first.fill(LOCATION)
                        log(f"  Typed into: {sel}")
                        typed = True
                        break
                except Exception:
                    continue
            if not typed:
                log("  No input found — aborting")
                return []

            page.wait_for_timeout(2000)

            # 4. Select from autocomplete / dropdown
            select_clicked = False
            try:
                page.wait_for_selector("button:has-text('Select')", timeout=3000)
                btns = page.locator("button:has-text('Select')")
                if btns.count() > 0:
                    btns.first.scroll_into_view_if_needed()
                    btns.first.evaluate("el => el.click()")
                    log("  Selected first dropdown result")
                    select_clicked = True
                    page.wait_for_timeout(1500)
            except Exception:
                pass

            if not select_clicked:
                log("  ArrowDown x2 + Enter fallback")
                page.keyboard.press("ArrowDown")
                page.wait_for_timeout(400)
                page.keyboard.press("ArrowDown")
                page.wait_for_timeout(400)
                page.keyboard.press("Enter")
                page.wait_for_timeout(1500)

            # Continue button
            for sel in ["button:has-text('Continue')", "button:has-text('Search')",
                        "button[type='submit']", "#btnContinue"]:
                try:
                    btn = page.locator(sel)
                    if btn.count() > 0 and btn.first.is_visible(timeout=3000):
                        btn.first.click()
                        log(f"  Continue: {sel}")
                        break
                except Exception:
                    continue

            page.wait_for_load_state("domcontentloaded")
            page.wait_for_timeout(1500)

            # 5. Fill dates
            log(f"Setting dates: {PICKUP_DATE} -> {RETURN_DATE}")
            pickup_inp = return_inp = None
            for sel in ["input#pickupDate", "input#from-date",
                        "input[id*='pickup'][id*='date' i]",
                        "input[placeholder*='Pick-up' i]"]:
                try:
                    loc = page.locator(sel)
                    if loc.count() > 0 and loc.first.is_visible(timeout=1500):
                        pickup_inp = loc.first
                        log(f"  Pickup field: {sel}")
                        break
                except Exception:
                    continue
            for sel in ["input#returnDate", "input#to-date",
                        "input[id*='return'][id*='date' i]",
                        "input[placeholder*='Return' i]"]:
                try:
                    loc = page.locator(sel)
                    if loc.count() > 0 and loc.first.is_visible(timeout=1500):
                        return_inp = loc.first
                        log(f"  Return field: {sel}")
                        break
                except Exception:
                    continue

            if pickup_inp and return_inp:
                _fill_date(page, pickup_inp, PICKUP_DATE)
                _fill_date(page, return_inp, RETURN_DATE)
                log("  Dates filled")
            else:
                log("  Date fields not found — proceeding anyway")

            page.wait_for_timeout(2500)

            # 6. Browse vehicles
            for sel in ["button:has-text('Reserve')", "button:has-text('Browse Vehicles')",
                        "button:has-text('Search')", "button[type='submit']"]:
                try:
                    btn = page.locator(sel)
                    if btn.count() > 0 and btn.first.is_visible(timeout=3000):
                        btn.first.click()
                        log(f"  Browse: {sel}")
                        break
                except Exception:
                    continue

            # 7. Wait for results
            log("Waiting for results page...")
            try:
                page.wait_for_load_state("domcontentloaded", timeout=15000)
            except Exception:
                pass
            page.wait_for_timeout(2500)
            page.screenshot(path=os.path.join(OUTPUT_DIR, "results.png"))

            # 8. Scroll to trigger lazy-load
            log("Scrolling to load all vehicles...")
            for _ in range(12):
                page.evaluate("window.scrollBy({top: 300, left: 0, behavior: 'smooth'})")
                page.wait_for_timeout(700)
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(1000)

            # 9. Extract
            page_text = _truncate_enterprise_listing(page.inner_text("body"))
            with open(os.path.join(OUTPUT_DIR, "page_content.txt"), "w") as f:
                f.write(page_text)
            log(f"Page text: {len(page_text)} chars")

            log("Running parser...")
            cars = _extract_enterprise_cards(page_text)
            log("Running DOM extractor...")
            dom_cars = _extract_dom_enterprise(page)

            # Merge deduplicated
            seen = {}
            for c in cars + dom_cars:
                key = (c["car_name"].lower(), c["price_per_day"].lower())
                if key not in seen:
                    seen[key] = c
            cars = list(seen.values())
            log(f"Merged: {len(cars)} unique cars")

            cars = _sanity_filter(cars)
            log(f"After sanity filter: {len(cars)} cars")

            cars = _finalize_cars_with_ollama(cars, page_text, SITE_HINT)
            cars = _sanity_filter(cars)
            log(f"Final: {len(cars)} valid cars")

        except Exception as e:
            log(f"ERROR: {e}")
            import traceback; traceback.print_exc()
            try:
                page.screenshot(path=os.path.join(OUTPUT_DIR, "error.png"))
            except Exception:
                pass
        finally:
            browser.close()
    return cars


def save(cars):
    if not cars:
        log("No cars to save.")
        return
    stamp = now_mst().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(cars)
    extra_cols = [c for c in df.columns if c not in CSV_COLUMNS]
    df = df[[c for c in CSV_COLUMNS if c in df.columns] + extra_cols]
    csv_path  = os.path.join(OUTPUT_DIR, f"{stamp}.csv")
    json_path = os.path.join(OUTPUT_DIR, f"{stamp}.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(cars, f, indent=2, ensure_ascii=False)
    print(f"\\n{'='*58}")
    print(f"  {len(cars)} vehicles  |  {now_mst().strftime('%Y-%m-%d %H:%M:%S MST')}")
    print(f"  Location : {LOCATION}")
    print(f"  CSV      : {csv_path}")
    print(f"  JSON     : {json_path}")
    print(f"{'='*58}")
    print("\\nFirst 5 results:")
    for i, c in enumerate(cars[:5], 1):
        print(f"  {i}. {c['car_name']} ({c['car_type']}) — {c['price_per_day']}")


if __name__ == "__main__":
    log(f"Starting Enterprise scraper for: {LOCATION}")
    log(f"Dates: {PICKUP_DATE} -> {RETURN_DATE}")
    cars = scrape()
    save(cars)
'''

    # ── Sixt body ────────────────────────────────────────────────────────────
    elif "sixt" in site.lower():
        sixt_header = textwrap.dedent(f"""\
#!/usr/bin/env python3
\"\"\"
{'='*64}
  Site     : {site}
  Location : {location}
  Generated: {generated_at}
  Pickup   : {pickup_iso}
  Return   : {return_iso}
{'='*64}
Run directly:
    python3 {stamp}.py

Optional AI (Ollama — same prompts as run.py):
    ollama serve && ollama pull qwen2.5:7b
    USE_OLLAMA=0 python3 {stamp}.py    # parsers only
    AI_MODE=merge python3 {stamp}.py   # parser + Ollama merge
\"\"\"

import os, re, json, logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from playwright.sync_api import sync_playwright, Page

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

LOCATION        = "{safe_loc}"
SITE_HINT       = "{site_hint}"
PICKUP_TIME     = "10:00"
RETURN_TIME     = "10:00"
HEADLESS        = False
USE_OLLAMA      = os.environ.get("USE_OLLAMA", "1") not in ("0", "", "false", "False")
AI_MODE         = os.environ.get("AI_MODE", "hybrid").lower()
OLLAMA_BASE     = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
EXTRACTOR_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
MST             = ZoneInfo("America/Edmonton")

def now_mst():
    return datetime.now(MST)

def mst_log_prefix():
    return now_mst().strftime("%H:%M:%S")

def log(msg):
    print(f"[{{mst_log_prefix()}}] {{msg}}")

OUTPUT_DIR = os.path.join(
    "scraper_outputs", "Sixt_Canada",
    re.sub(r"[^a-zA-Z0-9]", "_", LOCATION)[:28]
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_COLUMNS = [
    "pickup_date", "return_date", "pickup_time", "return_time",
    "car_name", "car_type", "price_per_day", "transmission", "seats", "bags", "location",
]

# ── regex / constants ────────────────────────────────────────────────────────
PRICE_RE = re.compile(r"((?:CA)?\\s*\\$\\s*[\\d,]+\\.?\\d*\\s*/\\s*day)", re.IGNORECASE)
NUM_RE   = re.compile(r"[\\d,]+\\.?\\d*")
CAR_TYPES = ["Economy","Compact","Intermediate","Standard","Fullsize","Premium",
             "Luxury","SUV","Minivan","Van","Convertible","Pickup","Wagon",
             "Crossover","Electric","Hybrid","Sedan","Elite","Truck"]
TYPE_RE  = re.compile(r"\\b(" + "|".join(CAR_TYPES) + r")\\b", re.IGNORECASE)
TRANS_RE = re.compile(r"\\b(Automatic|Manual)\\b", re.IGNORECASE)
SIXT_CLASS_LINE_RE = re.compile(
    r"^(ECONOMY|COMPACT|INTERMEDIATE|STANDARD|FULLSIZE|FULL-SIZE|PREMIUM|MINIVAN|LUXURY|VAN|SUV|CONVERTIBLE)"
    r"(\\s+ELITE)?\\s+\\(([^)]+)\\)\\s*$", re.IGNORECASE
)

# ── popup dismissers ─────────────────────────────────────────────────────────

def _dismiss_popups(page):
    for sel in ["button:has-text('I AGREE')", "button:has-text('Accept all')",
                "button:has-text('Accept')", "button:has-text('Close')",
                "#onetrust-accept-btn-handler", "button[aria-label='Close']"]:
        try:
            loc = page.locator(sel)
            if loc.count() > 0 and loc.first.is_visible(timeout=1500):
                loc.first.click(timeout=3000)
                log(f"  Popup dismissed: {{sel}}")
                page.wait_for_timeout(1200)
                return
        except Exception:
            continue

""")
        body = sixt_header + _SIXT_PARSER_SRC + _sixt_metadata_helpers_src() + _ollama_script_src() + _SIXT_SCRIPT_TAIL

    # ── Fallback ─────────────────────────────────────────────────────────────
    else:
        body = HEADER + '''\
if __name__ == "__main__":
    log("Generic site — extend this script with your scraping logic.")
'''

    script_path = os.path.join(folder, f"{stamp}.py")
    with open(script_path, "w") as f:
        f.write(body)
    os.chmod(script_path, 0o755)
    return script_path

def _dismiss_popups(page: Page, log_fn=None):
    if log_fn is None:
        log_fn = print
    def consent_visible() -> bool:
        try:
            return bool(page.evaluate("""
                () => {
                    const body = document.body.innerText || '';
                    if (!/Privacy Settings|Privacy & Cookie Policy|Marketing & Analytics/i.test(body)) {
                        return false;
                    }
                    return Array.from(document.querySelectorAll('button'))
                        .some(btn => /I\\s+AGREE|Accept all|Save Services/i.test((btn.innerText || '').trim()));
                }
            """))
        except Exception:
            return False

    def click_usercentrics_agree():
        try:
            return page.evaluate("""
                () => {
                    const visible = el => {
                        const r = el.getBoundingClientRect();
                        const s = window.getComputedStyle(el);
                        return r.width > 4 && r.height > 4
                            && s.display !== 'none'
                            && s.visibility !== 'hidden'
                            && s.opacity !== '0';
                    };
                    const labels = [/^I\\s+AGREE$/i, /^Accept all$/i, /^Agree$/i, /^Save Services$/i];
                    const buttons = Array.from(document.querySelectorAll('button')).filter(visible);
                    for (const label of labels) {
                        const btn = buttons.find(b => label.test((b.innerText || '').trim()));
                        if (btn) {
                            btn.scrollIntoView({block: 'center', inline: 'center'});
                            btn.click();
                            return (btn.innerText || '').trim();
                        }
                    }
                    return null;
                }
            """)
        except Exception:
            return None

    SELECTORS = [
        "button:has-text('I AGREE')", "button:has-text('I Agree')",
        "button:has-text('Agree')", "button:has-text('AGREE')",
        "button:has-text('Accept all')", "button:has-text('Accept')",
        "button:has-text('ACCEPT')", "button:has-text('Allow all')",
        "button:has-text('Got it')",
        "button:has-text('Save Services')",
        "button:has-text('Close')", "#onetrust-accept-btn-handler",
        "[data-testid*='accept'] button", "[data-testid*='consent'] button",
        ".iubenda-cs-accept-btn",
        "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
        "button[aria-label='Close']", "button[aria-label*='agree' i]",
        "button[aria-label*='accept' i]", "[class*='cookie'] button",
        "[class*='consent'] button", "[class*='uc-'] button",
    ]
    WORDS = {"agree", "accept", "allow", "got it", "continue"}
    deadline_ms = 10000
    started = now_mst()

    while (now_mst() - started).total_seconds() * 1000 < deadline_ms:
        clicked_label = click_usercentrics_agree()
        if clicked_label:
            page.wait_for_timeout(1500)
            if not consent_visible():
                log_fn(f"  Popup dismissed via Usercentrics button: {clicked_label}")
                return True
            log_fn(f"  Usercentrics click did not close modal yet: {clicked_label}")

        for sel in SELECTORS:
            try:
                loc = page.locator(sel)
                count = min(loc.count(), 4)
                for i in range(count):
                    btn = loc.nth(i)
                    if btn.is_visible(timeout=700):
                        try:
                            btn.click(timeout=2500, force=True)
                        except Exception:
                            btn.evaluate("el => el.click()")
                        page.wait_for_timeout(1200)
                        if not consent_visible():
                            log_fn(f"  Popup dismissed: {sel}")
                            return True
            except Exception:
                continue

        try:
            for btn in page.locator("button").all():
                try:
                    txt = btn.inner_text(timeout=300).strip().lower()
                    if any(w in txt for w in WORDS) and btn.is_visible(timeout=300):
                        try:
                            btn.click(timeout=2500, force=True)
                        except Exception:
                            btn.evaluate("el => el.click()")
                        page.wait_for_timeout(1200)
                        if not consent_visible():
                            log_fn(f"  Brute-force popup: '{txt}'")
                            return True
                except Exception:
                    continue
        except Exception:
            pass

        page.wait_for_timeout(500)

    log_fn("  No popup dismissed after waiting")
    return False
def _dismiss_enterprise_promos(page: Page, log_fn) -> None:
    """
    Close Enterprise-specific promotional overlays that appear after page load
    (e.g. the 'On Every Corner' tournament banner, the 15%-off strip, etc.)
    before we attempt to interact with the location input.
    """

    log_fn("   Waiting 5 s for all promo banners to load...")
    page.wait_for_timeout(5000)
    
    PROMO_CLOSE_SELS = [
        "button[aria-label='Close']",
        "button[aria-label='close']",
        "[class*='modal'] button[aria-label*='lose' i]",
        "[class*='overlay'] button[aria-label*='lose' i]",
        "[class*='dialog'] button[aria-label*='lose' i]",
        "[class*='modal'] .close",
        "[class*='modal'] [class*='close']",
        "[class*='dialog'] [class*='close']",
        "[class*='overlay'] [class*='close']",
    ]

    dismissed = False
    for sel in PROMO_CLOSE_SELS:
        try:
            loc = page.locator(sel)
            if loc.count() > 0 and loc.first.is_visible(timeout=1500):
                loc.first.click(timeout=3000)
                log_fn(f"   Promo modal closed via: {sel}")
                page.wait_for_timeout(1000)
                dismissed = True
                break
        except Exception:
            continue

  
    if not dismissed:
        try:
            page.mouse.click(1367, 288)
            log_fn("   Promo modal closed via coordinate click on ×")
            page.wait_for_timeout(1000)
            dismissed = True
        except Exception:
            pass


    if not dismissed:
        try:
            page.keyboard.press("Escape")
            log_fn("   Sent Escape to dismiss modal")
            page.wait_for_timeout(800)
        except Exception:
            pass

    try:
        if not dismissed:
            removed = page.evaluate("""
                () => {
                    const removed = [];
                    const sels = [
                        '[role="dialog"]', '[aria-modal="true"]',
                        '[class*="overlay"]', '[class*="Overlay"]',
                    ];
                    for (const sel of sels) {
                        for (const el of document.querySelectorAll(sel)) {
                            const style = window.getComputedStyle(el);
                            const rect = el.getBoundingClientRect();
                            if (rect.width > 300 && rect.height > 200
                                    && style.display !== 'none'
                                    && style.visibility !== 'hidden'
                                    && style.position === 'fixed') {
                                el.style.display = 'none';
                                removed.push(sel);
                            }
                        }
                    }
                    return removed;
                }
            """)
            if removed:
                log_fn(f"   JS force-hidden overlays: {removed}")
                page.wait_for_timeout(500)
    except Exception:
        pass

    # ── 6. Final short wait so the page settles ──────────────────────────────
    page.wait_for_timeout(1500)
    log_fn("   All Enterprise promo banners handled — safe to type")
    

def _click_show_cars(page: Page, log_fn) -> bool:
    """Clicks the orange 'Show cars' Sixt button."""
    SHOW_CARS_SELS = [
        "button:has-text('Show cars')", "button:has-text('SHOW CARS')",
        "button:has-text('Show Cars')", "[data-testid='show-cars-button']",
        "button:has-text('Show stations')", "button:has-text('SHOW STATIONS')",
        "button:has-text('Show Stations')", "[data-testid*='show-stations']",
        "[data-testid*='showStations']",
        "[data-testid*='show-cars']", "[data-testid*='showCars']",
        "button[class*='primary']", "button[class*='cta']",
        "button[class*='search-btn']", "button[class*='searchBtn']",
        "form button[type='submit']", "button[type='submit']",
    ]
    for sel in SHOW_CARS_SELS:
        try:
            btn = page.locator(sel)
            if btn.count() > 0 and btn.first.is_visible(timeout=1500):
                btn.first.scroll_into_view_if_needed()
                page.wait_for_timeout(200)
                btn.first.click()
                log_fn(f"   'Show cars' -> {sel}")
                return True
        except Exception:
            continue

    
    try:
        result = page.evaluate("""
            () => {
                const btns = Array.from(document.querySelectorAll('button'));
                for (const btn of btns) {
                    const txt = btn.innerText.trim().toLowerCase();
                    if (txt.includes('show') || txt.includes('car') || txt.includes('search')) {
                        const style = window.getComputedStyle(btn);
                        const bg = style.backgroundColor;
                        if (bg && bg !== 'transparent' && bg !== 'rgba(0, 0, 0, 0)') {
                            btn.click();
                            return btn.innerText.trim();
                        }
                    }
                }
                return null;
            }
        """)
        if result:
            log_fn(f"   'Show cars' JS scan -> '{result}'")
            return True
    except Exception:
        pass


    if _smart_click(
        page,
        "the primary button that submits the search to show stations, pickup branches, or available rental cars",
        log_fn=log_fn,
    ):
        return True

    log_fn("   WARNING: 'Show cars/stations' button not found")
    return False


def _sixt_is_branch_page(page: Page) -> bool:
    try:
        if "nearbybranches" in page.url.lower():
            return True
        body = page.inner_text("body")[:2500].lower()
        return "select a pickup branch" in body or "pickup branch" in body
    except Exception:
        return False


def _sixt_first_branch_name(page: Page) -> str | None:
    """First available branch title from the left list (not fully booked)."""
    try:
        return page.evaluate("""
            () => {
                const skip = /fully booked/i;
                const lines = document.body.innerText.split('\\n').map(l => l.trim()).filter(Boolean);
                let pastHeader = false;
                for (let i = 0; i < lines.length; i++) {
                    const line = lines[i];
                    if (/select a pickup branch/i.test(line)) {
                        pastHeader = true;
                        continue;
                    }
                    if (!pastHeader) continue;
                    if (skip.test(line)) continue;
                    if (/^(starting at|ca\\$|keyboard|map data|help|log in)/i.test(line)) continue;
                    if (line.length < 5 || line.length > 55) continue;
                    if (/\\d{3,}.*\\b(NE|NW|SE|SW|Dr|St|Ave|Way|Rd|Blvd)\\b/i.test(line)) continue;
                    const nxt = (lines[i + 1] || '').toLowerCase();
                    const nxt2 = (lines[i + 2] || '').toLowerCase();
                    if (nxt.includes('starting at') || /^\\d+\\s/.test(nxt)
                        || nxt2.includes('starting at')) {
                        return line;
                    }
                }
                return null;
            }
        """)
    except Exception:
        return None


def _sixt_select_available_branch(page: Page, log_fn) -> bool:
    """Expand/select the first available branch in the left sidebar."""
    page.wait_for_timeout(1500)
    branch_name = _sixt_first_branch_name(page)
    if branch_name:
        log_fn(f"   First branch: {branch_name}")

    # Strategy 1: click branch title in left sidebar (smallest matching element)
    if branch_name:
        try:
            clicked = page.evaluate("""
                (name) => {
                    const skip = /fully booked/i;
                    let best = null;
                    let bestArea = Infinity;
                    for (const el of document.querySelectorAll('*')) {
                        const t = (el.innerText || '').trim();
                        if (t !== name) continue;
                        if (skip.test(el.closest('[class]')?.innerText || '')) continue;
                        const r = el.getBoundingClientRect();
                        if (r.width < 5 || r.height < 5 || r.top < 60) continue;
                        if (r.left > 520) continue;
                        const area = r.width * r.height;
                        if (area < bestArea) {
                            bestArea = area;
                            best = el;
                        }
                    }
                    if (best) {
                        best.click();
                        return name;
                    }
                    return null;
                }
            """, branch_name)
            if clicked:
                log_fn(f"   Branch title clicked: {clicked}")
                page.wait_for_timeout(2000)
                return True
        except Exception:
            pass

        try:
            loc = page.get_by_text(branch_name, exact=True)
            if loc.count() > 0:
                loc.first.scroll_into_view_if_needed()
                loc.first.click(timeout=5000)
                log_fn(f"   Branch via get_by_text: {branch_name}")
                page.wait_for_timeout(2000)
                return True
        except Exception:
            pass

    try:
        picked = page.evaluate("""
            () => {
                const skip = /fully booked/i;
                const rows = [];
                for (const el of document.querySelectorAll(
                    'div, li, article, button, [role="button"], [data-testid*="branch"]'
                )) {
                    const t = (el.innerText || '').trim();
                    if (!t || t.length > 400 || t.length < 15) continue;
                    if (skip.test(t)) continue;
                    if (!/starting at/i.test(t) && !/CA\\$[\\d,.]+\\s*\\/\\s*day/i.test(t)) continue;
                    const r = el.getBoundingClientRect();
                    if (r.left > 480 || r.width < 100 || r.top < 80) continue;
                    rows.push({el, top: r.top, t: t.split('\\n')[0].slice(0, 50)});
                }
                rows.sort((a, b) => a.top - b.top);
                if (rows.length) {
                    rows[0].el.click();
                    return rows[0].t;
                }
                return null;
            }
        """)
        if picked:
            log_fn(f"   Branch row clicked: {picked}")
            page.wait_for_timeout(2000)
            return True
    except Exception:
        pass


    for sel in [
        "[class*='branch'] button",
        "[class*='station'] button",
        "[data-testid*='branch']",
        "[aria-expanded='false']",
    ]:
        try:
            el = page.locator(sel)
            if el.count() > 0 and el.first.is_visible(timeout=1000):
                el.first.click()
                log_fn(f"   Branch expand: {sel}")
                page.wait_for_timeout(2000)
                return True
        except Exception:
            continue

    log_fn("   WARNING: could not select a branch")
    return False


def _sixt_wait_for_show_offers(page: Page, timeout_ms: int = 12000) -> bool:
    try:
        page.wait_for_function("""
            () => Array.from(document.querySelectorAll('button, a, [role="button"]'))
                .some(el => /show\\s+offers?/i.test((el.innerText || '').trim()))
        """, timeout=timeout_ms)
        return True
    except Exception:
        return False


def _sixt_click_show_offers(page: Page, log_fn) -> bool:
    """After a branch is selected, open the car offers list."""
    _sixt_wait_for_show_offers(page, timeout_ms=8000)
    page.wait_for_timeout(500)

    for sel in [
        "button:has-text('Show offers')",
        "button:has-text('SHOW OFFERS')",
        "button:has-text('Show offer')",
        "a:has-text('Show offers')",
        "a:has-text('SHOW OFFERS')",
        "[data-testid*='show-offers']",
        "[data-testid*='showOffers']",
        "button:has-text('Show cars')",
    ]:
        try:
            btn = page.locator(sel)
            for i in range(min(btn.count(), 3)):
                item = btn.nth(i)
                if item.is_visible(timeout=1500):
                    item.scroll_into_view_if_needed()
                    item.click(timeout=5000)
                    log_fn(f"   Show offers -> {sel} [{i}]")
                    page.wait_for_timeout(5000)
                    return True
        except Exception:
            continue

    try:
        label = page.evaluate("""
            () => {
                const els = Array.from(document.querySelectorAll(
                    'button, a, [role="button"], input[type="button"], input[type="submit"]'
                ));
                for (const el of els) {
                    const t = (el.innerText || el.value || '').trim();
                    if (!/show\\s+offers?/i.test(t)) continue;
                    const r = el.getBoundingClientRect();
                    if (r.width < 5 || r.height < 5) continue;
                    el.scrollIntoView({block: 'center'});
                    el.click();
                    return t;
                }
                return null;
            }
        """)
        if label:
            log_fn(f"   Show offers JS -> '{label}'")
            page.wait_for_timeout(5000)
            return True
    except Exception:
        pass

    if _smart_click(
        page,
        "the button or link that opens the list of car offers for the selected branch",
        log_fn=log_fn,
    ):
        page.wait_for_timeout(5000)
        return True

    log_fn("   WARNING: Show offers button not found")
    return False


def _sixt_branch_to_offers(page: Page, log_fn) -> bool:
    """Branch picker -> select station -> Show offers -> car list."""
    if not _sixt_is_branch_page(page):
        return False

    log_fn("Branch selection page detected")
    page.screenshot(path="/tmp/sixt_stations.png")
    log_fn("   /tmp/sixt_stations.png")

    offers_ok = False
    for attempt in range(1, 4):
        log_fn(f"   Branch/offers attempt {attempt}/3...")
        _sixt_select_available_branch(page, log_fn)
        page.wait_for_timeout(1500)
        page.screenshot(path=f"/tmp/sixt_branch_try_{attempt}.png")

        if _sixt_click_show_offers(page, log_fn):
            offers_ok = True
            break

       
        branch_name = _sixt_first_branch_name(page)
        if branch_name:
            try:
                page.get_by_text(branch_name, exact=True).first.click(timeout=3000)
                page.wait_for_timeout(1500)
            except Exception:
                pass

    if not offers_ok:
        log_fn("   All Show offers attempts failed")
        return False

    try:
        page.wait_for_load_state("domcontentloaded", timeout=20000)
    except Exception:
        pass
    try:
        page.wait_for_function(
            "() => /which car do you want|CA\\$[\\d,.]+\\/\\s*day/i.test(document.body.innerText)",
            timeout=15000,
        )
        log_fn("   Car offers page loaded")
    except Exception:
        log_fn("   Car offers page wait timed out — continuing")
    page.wait_for_timeout(3000)
    page.screenshot(path="/tmp/sixt_results.png")
    log_fn("   /tmp/sixt_results.png")
    return True



def _sixt_location_error_visible(page: Page) -> bool:
    """True while Sixt is still asking the user to pick a real pickup location."""
    try:
        return "please select a pickup location" in page.inner_text("body", timeout=1500).lower()
    except Exception:
        return False


_SIXT_SELECT_BEST_JS = r"""
({query, requirePrimary}) => {
  const q = (query || '').toLowerCase();
  const words = q.split(/\s+/).filter(Boolean);
  const skip = new Set(['airport', 'international', 'intl', 'int', 'city', 'the']);
  const primary = words.find(w => !skip.has(w)) || words[0] || '';
  const sels = '[data-testid*="suggestion"],[class*="suggestion"],[role="option"],'
             + '[class*="result"] li,[class*="autocomplete"] li,ul[role="listbox"] li,'
             + '[id*="downshift"] li,[id*="react-select"] [role="option"],li';
  const visible = el => {
    const r = el.getBoundingClientRect();
    const s = window.getComputedStyle(el);
    return r.width > 2 && r.height > 2 && s.display !== 'none'
        && s.visibility !== 'hidden' && s.opacity !== '0';
  };
  let items = Array.from(document.querySelectorAll(sels))
    .filter(visible)
    .filter(el => (el.innerText || '').trim().length > 2);
  if (!items.length) return {status: 'no_items'};

  // Prefer leaf rows: drop any element that contains another candidate, so we
  // click a single suggestion line and not the whole dropdown container.
  const set = new Set(items);
  const leaves = items.filter(el => {
    for (const other of set) {
      if (other !== el && el.contains(other)) return false;
    }
    return true;
  });
  const pool = leaves.length ? leaves : items;

  let best = null, bestScore = -Infinity;
  for (const el of pool) {
    const raw = (el.innerText || '').trim();
    const t = raw.toLowerCase();
    let score = 0;
    for (const w of words) if (t.includes(w)) score += 1;
    if (primary && t.includes(primary)) score += 5; else if (primary) score -= 10;
    if (/\bairport\b/.test(t)) score += 1.5;
    if (el.getAttribute('role') === 'option') score += 2;
    if ((el.getAttribute('data-testid') || '').toLowerCase().includes('suggestion')) score += 2;
    // Favour short, specific rows over long container blocks.
    score -= raw.length / 50;
    if (score > bestScore) { bestScore = score; best = el; }
  }
  if (!best) return {status: 'no_match'};
  if (requirePrimary && primary && !(best.innerText || '').toLowerCase().includes(primary)) {
    return {status: 'no_primary'};
  }

  best.scrollIntoView({block: 'center', inline: 'center'});
  const r = best.getBoundingClientRect();
  const opts = {bubbles: true, cancelable: true, view: window,
                clientX: r.left + r.width / 2, clientY: r.top + r.height / 2};
  for (const type of ['pointerover', 'pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click']) {
    best.dispatchEvent(new MouseEvent(type, opts));
  }
  return {status: 'clicked',
          text: (best.innerText || '').trim().replace(/\s+/g, ' ').slice(0, 100)};
}
"""

_LOC_SKIP_WORDS = {"airport", "international", "intl", "int", "city", "the"}


def _primary_token(text: str) -> str:
    words = [w for w in re.split(r"\s+", (text or "").lower()) if w]
    return next((w for w in words if w not in _LOC_SKIP_WORDS), words[0] if words else "")


def _sixt_commit_location(page: Page, location: str, log_fn) -> bool:
    """Type the location with real keystrokes and actually select a suggestion.

    Sixt only accepts the location once a dropdown suggestion is chosen; just
    filling the input leaves a 'Please select a pickup location' error and the
    'Show cars' button does nothing. We wait for a suggestion that genuinely
    matches the city (not a fuzzy guess like YVR for YYC), click it, and verify.
    """
    box = None
    for sel in [
        'input[placeholder*="Airport" i]',
        'input[placeholder*="city" i]',
        'input[placeholder*="location" i]',
        'input[placeholder*="pickup" i]',
        'input[type="search"]',
    ]:
        try:
            loc = page.locator(sel)
            if loc.count() > 0 and loc.first.is_visible(timeout=3000):
                box = loc.first
                break
        except Exception:
            continue
    if box is None:
        try:
            cand = page.locator("input:visible").first
            if cand.count() > 0 and cand.is_visible(timeout=2500):
                box = cand
        except Exception:
            box = None
    if box is None:
        log_fn("         Could not find the location search box.")
        return False

    primary = _primary_token(location)

    def committed() -> bool:
        try:
            body = page.inner_text("body", timeout=1200).lower()
        except Exception:
            body = ""
        return "please select a pickup location" not in body

    def select_best() -> str | None:
        """Wait for a city-matching suggestion, then click it atomically."""
        # First pass: insist on a suggestion that contains the city name.
        for _ in range(16):
            try:
                res = page.evaluate(_SIXT_SELECT_BEST_JS,
                                    {"query": location, "requirePrimary": True})
            except Exception:
                res = {"status": "error"}
            if res and res.get("status") == "clicked":
                return res.get("text")
            page.wait_for_timeout(250)
        # Second pass: accept the best available even without an exact city match.
        try:
            res = page.evaluate(_SIXT_SELECT_BEST_JS,
                                {"query": location, "requirePrimary": False})
            if res and res.get("status") == "clicked":
                return res.get("text")
        except Exception:
            pass
        return None

    for attempt in range(1, 4):
        try:
            box.click()
            box.fill("")
            page.wait_for_timeout(300)
            try:
                box.press_sequentially(location, delay=70)
            except Exception:
                box.type(location, delay=70)
        except Exception:
            pass

        page.wait_for_timeout(800)
        clicked = select_best()
        page.wait_for_timeout(1000)

    
        good_city = (not primary) or (clicked and primary in clicked.lower())
        if clicked and committed() and good_city:
            log_fn(f"         Location accepted: {clicked}")
            return True

        page.keyboard.press("ArrowDown")
        page.wait_for_timeout(300)
        page.keyboard.press("Enter")
        page.wait_for_timeout(1200)
        if committed() and ((not primary) or attempt >= 2):
            log_fn(f"         Location accepted: {clicked or location}")
            return True

        log_fn(f"         Location not committed yet — retrying ({attempt}/3)")

    return committed()


def run_sixt(
    location: str,
    headless: bool = False,
    logs: list = None,
    pickup_date: str | None = None,
    return_date: str | None = None,
    pickup_time: str | None = None,
    return_time: str | None = None,
) -> list:
    """Scrape Sixt.ca for car rental prices.

    Uses Sixt's preloaded pickup/return dates and times from the live search
    form (typically pickup ~2 days out, return ~5 days out). CLI date/time
    args override the page only when you pass them explicitly.
    """
    if logs is None:
        logs = []

    def log(msg):
        entry = f"[{mst_log_prefix()}] {msg}"
        logs.append(entry)
        print(entry)

    user_meta: dict[str, str] = {}
    if pickup_date:
        user_meta["pickup_date"] = _to_iso_date(pickup_date)
    if return_date:
        user_meta["return_date"] = _to_iso_date(return_date)
    if pickup_time:
        user_meta["pickup_time"] = _normalize_time_str(pickup_time)
    if return_time:
        user_meta["return_time"] = _normalize_time_str(return_time)

    rental_meta: dict[str, str] = {}
    cars = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = browser.new_page()
        try:
            log("Step 1/5  Opening Sixt.ca")
            page.goto("https://www.sixt.ca", wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(3000)
            page.screenshot(path="/tmp/sixt_01_loaded.png")

            log("Step 2/5  Accepting the cookie banner")
            _dismiss_popups(page, log)

            _wait_for_sixt_search_form(page)
            rental_meta = _read_sixt_rental_metadata(page)

            search_term = resolve_iata(location)
            if search_term != location:
                log(f"Looking up airport code {location.strip().upper()} -> {search_term}")
            log(f"Step 3/5  Searching for location: {search_term}")
            committed = _sixt_commit_location(page, search_term, log)
            page.wait_for_timeout(1000)
            page.screenshot(path="/tmp/sixt_02_location.png")
            if not committed:
                log("         Sixt still wants a pickup location — check /tmp/sixt_02_location.png")
                return []

            rental_meta = _accumulate_rental_metadata(rental_meta, _read_sixt_rental_metadata(page))
            if rental_meta.get("pickup_date") or rental_meta.get("return_date"):
                log(
                    "         Preloaded search dates: "
                    f"{rental_meta.get('pickup_date', '?')} {rental_meta.get('pickup_time', '')} -> "
                    f"{rental_meta.get('return_date', '?')} {rental_meta.get('return_time', '')}".rstrip()
                )

            log("Step 4/5  Loading available cars (using Sixt's preloaded dates)")
            cars_clicked = _click_show_cars(page, log)
            if cars_clicked:
                page.wait_for_timeout(1500)
                if _sixt_location_error_visible(page):
                    log("         Sixt rejected the typed location — selecting the dropdown result again")
                    if not _sixt_commit_location(page, location, log):
                        page.screenshot(path="/tmp/sixt_02_location.png")
                        log("         Location still not accepted — check /tmp/sixt_02_location.png")
                        return []
                    cars_clicked = _click_show_cars(page, log)
                if not cars_clicked:
                    log("         Couldn't open the car list after location retry.")
                    return []
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=20000)
                except Exception:
                    pass
                page.wait_for_timeout(4000)
            else:
                log("         Couldn't open the car list — check /tmp/sixt_02_location.png")
                return []

            if _sixt_is_branch_page(page):
                log("         Choosing the pickup branch")
                _sixt_branch_to_offers(page, log)
            elif "station" in (page.inner_text("body")[:1500].lower()):
                log("         Choosing the pickup branch")
                _sixt_branch_to_offers(page, log)

            last_h = page.evaluate("document.body.scrollHeight")
            for i in range(15):
                page.evaluate("window.scrollBy(0, 800)")
                page.wait_for_timeout(800)
                new_h = page.evaluate("document.body.scrollHeight")
                if i > 5 and new_h == last_h:
                    break
                last_h = new_h
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(2000)
            page.screenshot(path="/tmp/sixt_03_results.png")

            log("Step 5/5  Reading the car offers")
            page_text = page.inner_text("body")
            with open("/tmp/sixt_page_content.txt", "w") as f:
                f.write(page_text)

            price_hits = len(re.findall(r"CA\$[\d,]+\.?\d*/day", page_text, re.IGNORECASE))
            if price_hits == 0:
                log("         No prices on the page — it may not have loaded.")
                page.screenshot(path="/tmp/sixt_no_prices.png")

            if AI_FIRST:
                log("         Reading offers with Ollama (AI-first)")
                cars = _ollama_strict_extract(page_text, search_term, "Sixt.ca")
                parser_cars = _merge_sixt_by_price(
                    _extract_dom_sixt(page, search_term),
                    _extract_sixt_cards(page_text, search_term),
                )
                if AI_MERGE:
                    log("         AI-merge enabled — adding parser results")
                    log(f"         Parser found {len(parser_cars)} additional candidates")
                    cars = _merge_results(cars, parser_cars)
                elif cars and parser_cars:
                    log("         Backfilling transmission/seats/bags from parser")
                    cars = _backfill_from_parser(cars, parser_cars)
                    if len(cars) < len(parser_cars):
                        log(f"         Ollama found {len(cars)}/{len(parser_cars)} — adding parser offers AI missed")
                        cars = _merge_sixt_by_price(cars, parser_cars)
                elif not cars:
                    log("         AI returned nothing — using the page parser")
                    cars = parser_cars
            else:
                sixt_cars = _extract_sixt_cards(page_text, search_term)
                dom_sixt = _extract_dom_sixt(page, search_term)
                cars = _merge_sixt_by_price(dom_sixt, sixt_cars)

                if not cars:
                    log("         Page parser found nothing — trying generic regex parser")
                    cars = _regex_extract(page_text, search_term)
                if not cars:
                    log("         Page parser found nothing — asking the AI to read it")
                    cars = _ollama_strict_extract(page_text, search_term, "Sixt.ca")
                elif any(c.get("car_name") in (None, "", "Unknown") for c in cars):
                    log("         Filling in a few missing names with the AI")
                    cars = _ollama_enrich(cars, page_text, search_term, site_hint="Sixt.ca")

            cars = _sanity_filter(cars)
            rental_meta = _accumulate_rental_metadata(rental_meta, _read_sixt_rental_metadata(page))
            rental_meta = _finalize_rental_metadata(rental_meta, user_meta, site="sixt")
            log(
                f"         CSV dates/times: "
                f"{rental_meta.get('pickup_date')} {rental_meta.get('pickup_time', '')} -> "
                f"{rental_meta.get('return_date')} {rental_meta.get('return_time', '')}".rstrip()
            )
            cars = _stamp_rental_dates(
                cars,
                rental_meta["pickup_date"],
                rental_meta["return_date"],
                rental_meta.get("pickup_time", ""),
                rental_meta.get("return_time", ""),
            )
            log(f"Done. Found {len(cars)} cars.")

        except Exception as e:
            log(f"Something went wrong: {e}")
            import traceback
            traceback.print_exc()
            try:
                page.screenshot(path="/tmp/sixt_error.png")
            except Exception:
                pass
        finally:
            browser.close()

    return cars


ENTERPRISE_LOCATION_SELECTORS = [
    "input#pickupLocationTextBox",
    "input#geoLocation",
    "input[name='location-search']",
    "input[id*='location' i]",
    "input[placeholder*='ZIP' i]",
    "input[placeholder*='city' i]",
    "input[placeholder*='airport' i]",
    "input[type='text']:visible",
]


def run_enterprise(
    location: str,
    headless: bool = False,
    logs: list = None,
    pickup_date: str | None = None,
    return_date: str | None = None,
    pickup_time: str | None = None,
    return_time: str | None = None,
) -> list:
    if logs is None:
        logs = []

    def log(msg):
        entry = f"[{mst_log_prefix()}] {msg}"
        logs.append(entry)
        print(entry)

    user_meta: dict[str, str] = {}
    if pickup_date:
        user_meta["pickup_date"] = _to_iso_date(pickup_date)
    if return_date:
        user_meta["return_date"] = _to_iso_date(return_date)
    if pickup_time:
        user_meta["pickup_time"] = _normalize_time_str(pickup_time)
    if return_time:
        user_meta["return_time"] = _normalize_time_str(return_time)

    cars = []

    search_term = resolve_iata(location)
    if search_term != location:
        log(f"Looking up airport code {location.strip().upper()} -> {search_term}")

    display_name = search_term
    start_url = "https://www.enterprise.com/en/car-rental/locations.html"

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = browser.new_page()
        try:
            log("Step 1/6  Opening Enterprise.com")
            page.goto(start_url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(1500)

            log("Step 2/6  Closing pop-ups and promo banners")
            _dismiss_popups(page, log)
            _dismiss_enterprise_promos(page, log)

            log(f"Step 3/6  Searching for location: {search_term}")
            typed = False
            for sel in ENTERPRISE_LOCATION_SELECTORS:
                try:
                    inp = page.locator(sel)
                    if inp.count() > 0 and inp.first.is_visible(timeout=3000):
                        inp.first.click(timeout=3000)
                        inp.first.fill("")
                        page.wait_for_timeout(200)
                        inp.first.fill(search_term)
                        log(f"         Location typed via {sel}")
                        typed = True
                        break
                except Exception:
                    continue
            if not typed:
                typed = _smart_fill(
                    page,
                    "the location search box where you type the city or airport for car rental pickup",
                    search_term,
                    selectors=ENTERPRISE_LOCATION_SELECTORS,
                    log_fn=log,
                )
            if not typed:
                log("         Could not find the location search box — stopping.")
                return []

            page.wait_for_timeout(2000)
            page.screenshot(path="/tmp/enterprise_dropdown.png")

            select_clicked = False
            try:
                page.wait_for_selector("button:has-text('Select')", timeout=3000)
                btns = page.locator("button:has-text('Select')")
                if btns.count() > 0:
                    btns.first.scroll_into_view_if_needed()
                    page.wait_for_timeout(300)
                    btns.first.evaluate("el => el.click()")
                    select_clicked = True
                    page.wait_for_timeout(1500)
            except Exception:
                pass

            if not select_clicked:
                for kw in search_term.split() + ["Airport", "International"]:
                    try:
                        rows = page.locator(f"li:has-text('{kw}'), div:has-text('{kw}')")
                        if rows.count() > 0:
                            btn = rows.first.locator("button:has-text('Select')")
                            if btn.count() > 0:
                                btn.first.scroll_into_view_if_needed()
                                btn.first.evaluate("el => el.click()")
                                select_clicked = True
                                page.wait_for_timeout(1500)
                                break
                    except Exception:
                        continue

            if not select_clicked:
                page.keyboard.press("ArrowDown")
                page.wait_for_timeout(400)
                page.keyboard.press("ArrowDown")
                page.wait_for_timeout(400)
                page.keyboard.press("Enter")
                page.wait_for_timeout(1500)
            log("         Location selected")

            for sel in ["button:has-text('Continue')", "button:has-text('Search')",
                        "button[type='submit']", "#btnContinue"]:
                try:
                    btn = page.locator(sel)
                    if btn.count() > 0 and btn.first.is_visible(timeout=3000):
                        btn.first.click(timeout=5000)
                        break
                except Exception:
                    continue

            page.wait_for_load_state("domcontentloaded")
            page.wait_for_timeout(1500)

            if user_meta.get("pickup_date") and user_meta.get("return_date"):
                pickup_us = _iso_to_us(user_meta["pickup_date"])
                return_us = _iso_to_us(user_meta["return_date"])
                log(f"Step 4/6  Setting dates: {user_meta['pickup_date']} to {user_meta['return_date']}")
            else:
                pickup_us = (now_mst() + timedelta(days=1)).strftime("%m/%d/%Y")
                return_us = (now_mst() + timedelta(days=2)).strftime("%m/%d/%Y")
                log(f"Step 4/6  Setting dates: {pickup_us} to {return_us}")

            search_dates = {
                "pickup_date": user_meta.get("pickup_date") or _to_iso_date(pickup_us),
                "return_date": user_meta.get("return_date") or _to_iso_date(return_us),
            }

            def _fill(locator, value):
                try:
                    locator.scroll_into_view_if_needed()
                    locator.click()
                    page.wait_for_timeout(300)
                    locator.press("Control+a")
                    locator.type(value, delay=80)
                    page.wait_for_timeout(200)
                    locator.press("Tab")
                    return True
                except Exception:
                    return False

            pickup_inp = return_inp = None
            for sel in ["input#pickupDate", "input#from-date",
                        "input[id*='pickup'][id*='date' i]",
                        "input[placeholder*='Pick-up' i]"]:
                try:
                    loc = page.locator(sel)
                    if loc.count() > 0 and loc.first.is_visible(timeout=1500):
                        pickup_inp = loc.first
                        break
                except Exception:
                    continue

            for sel in ["input#returnDate", "input#to-date",
                        "input[id*='return'][id*='date' i]",
                        "input[placeholder*='Return' i]"]:
                try:
                    loc = page.locator(sel)
                    if loc.count() > 0 and loc.first.is_visible(timeout=1500):
                        return_inp = loc.first
                        break
                except Exception:
                    continue

            if pickup_inp and return_inp:
                _fill(pickup_inp, pickup_us)
                _fill(return_inp, return_us)
                log("         Dates set")
                page_dates = _read_enterprise_rental_metadata(page)
                search_dates = _accumulate_rental_metadata(search_dates, page_dates)
            else:
                log("         No date fields shown — using Enterprise's defaults")

            page.wait_for_timeout(2500)

            log("Step 5/6  Loading available vehicles")
            reserve_clicked = False
            for sel in ["button:has-text('Reserve')", "button:has-text('Browse Vehicles')",
                        "button:has-text('Search')", "button[type='submit']"]:
                try:
                    btn = page.locator(sel)
                    if btn.count() > 0 and btn.first.is_visible(timeout=3000):
                        btn.first.click()
                        reserve_clicked = True
                        break
                except Exception:
                    continue
            if not reserve_clicked:
                _smart_click(
                    page,
                    "the button that continues to reserve, browse, or view available vehicles",
                    log_fn=log,
                )

            try:
                page.wait_for_load_state("domcontentloaded", timeout=15000)
            except Exception:
                pass

            page.wait_for_timeout(2500)
            page.screenshot(path="/tmp/enterprise_results.png")

            try:
                alt = page.locator(
                    "text=/Explore Alternative|Hide [Aa]lternative/i"
                ).first
                if alt.is_visible(timeout=1500):
                    alt.scroll_into_view_if_needed()
                    page.wait_for_timeout(500)
            except Exception:
                pass

            _scroll_enterprise_results(page)

            page.evaluate("window.scrollTo({top: 0, left: 0, behavior: 'auto'})")
            page.wait_for_timeout(500)

            log("Step 6/6  Reading the vehicle offers")
            log("         Capturing page text")
            page_text = _truncate_enterprise_listing(page.inner_text("body"))
            with open("/tmp/enterprise_page_content.txt", "w") as f:
                f.write(page_text)
            log(f"         Page text captured ({len(page_text)} chars)")

            if AI_FIRST:
                log("         Reading offers with Ollama (AI-first)")
                cars = _ollama_strict_extract(page_text, display_name, site_hint="Enterprise.com")
                parser_cars = _merge_results(
                    _extract_enterprise_cards(page_text, display_name),
                    _extract_dom_enterprise(page, display_name),
                )
                if not parser_cars:
                    parser_cars = _regex_extract(page_text, display_name)
                if AI_MERGE:
                    log("         AI-merge enabled — adding parser results")
                    log(f"         Parser found {len(parser_cars)} additional candidates")
                    cars = _merge_results(cars, parser_cars)
                elif cars and parser_cars:
                    log("         Backfilling transmission/seats/bags from parser")
                    cars = _backfill_from_parser(cars, parser_cars)
                elif not cars:
                    log("         AI returned nothing — using the page parser")
                    cars = parser_cars
            else:
                log("         Running Enterprise page parser")
                ent_cars = _extract_enterprise_cards(page_text, display_name)
                log(f"         Page parser found {len(ent_cars)} vehicles")

                log("         Running Enterprise DOM parser")
                dom_ent = _extract_dom_enterprise(page, display_name)
                log(f"         DOM parser found {len(dom_ent)} vehicles")

                cars = _merge_results(ent_cars, dom_ent)

                if not cars:
                    log("         Parser found nothing — trying generic regex parser")
                    cars = _regex_extract(page_text, display_name)
                    log(f"         Generic regex parser found {len(cars)} vehicles")

                if not cars:
                    log("         Page parser found nothing — asking the AI to read it")
                    cars = _ollama_strict_extract(page_text, display_name, site_hint="Enterprise.com")
                elif any(c.get("car_name") in (None, "", "Unknown") for c in cars):
                    log("         Filling in a few missing names with the AI")
                    cars = _ollama_enrich(cars, page_text, display_name, site_hint="Enterprise.com")

            cars = _sanity_filter(cars)
            page_meta = _read_enterprise_rental_metadata(page)
            rental_meta = _merge_rental_metadata(search_dates, user_meta)
            if page_meta.get("pickup_time"):
                rental_meta["pickup_time"] = page_meta["pickup_time"]
            if page_meta.get("return_time"):
                rental_meta["return_time"] = page_meta["return_time"]
            log(
                f"         CSV dates/times: "
                f"{rental_meta.get('pickup_date')} {rental_meta.get('pickup_time', '')} -> "
                f"{rental_meta.get('return_date')} {rental_meta.get('return_time', '')}".rstrip()
            )
            cars = _stamp_rental_dates(
                cars,
                rental_meta["pickup_date"],
                rental_meta["return_date"],
                rental_meta.get("pickup_time", ""),
                rental_meta.get("return_time", ""),
            )
            log(f"Done. Found {len(cars)} vehicles.")

        except Exception as e:
            log(f"Something went wrong: {e}")
            try:
                page.screenshot(path="/tmp/enterprise_error.png")
            except Exception:
                pass
        finally:
            browser.close()

    return cars



def _plan_scrape_strategy(site_config):
    prompt = (f"You are a senior web scraping engineer.\n"
              f"Site URL: {site_config['url']}\nGoal: {site_config['goal']}\n"
              f"Search inputs: {json.dumps(site_config.get('search_params', {}))}\n\n"
              "Return a JSON object with keys:\n"
              "site_name, cookie_popup_selectors, search_steps, results_loaded_signal, "
              "pagination, extraction_hint. Each search_step has: action "
              "(type|click|press|wait), selector, value, description, wait_ms.")
    raw = _ollama_generate(prompt, model=PLANNER_MODEL, temperature=0.1)
    try:
        result = json.loads(raw)
    except Exception:
        result = {}
    if not result:
        try:
            clean = re.sub(r"```(?:json)?", "", raw).strip()
            s, e = clean.find("{"), clean.rfind("}")
            if s != -1 and e != -1:
                result = json.loads(clean[s:e + 1])
        except Exception:
            pass
    if not result:
        raise ValueError(f"Planner unparseable:\n{raw[:400]}")
    return result


class ScraperAgent:
    def __init__(self, site_config, headless=False):
        self.site_config = site_config
        self.headless    = headless
        self.logs: list  = []

    def _log(self, msg):
        entry = f"[{mst_log_prefix()}] {msg}"
        self.logs.append(entry)
        print(entry)

    def _run_generic(self):
        plan = _plan_scrape_strategy(self.site_config)
        self._log(f" Plan: {plan.get('site_name','Unknown')}")
        records = []
        FALLBACK_INPUTS = [
            'input[placeholder*="Airport" i]',
            'input[placeholder*="city" i]',
            'input[placeholder*="location" i]',
            'input[type="search"]',
        ]
        site_key = plan.get("site_name") or self.site_config.get("url", "generic")
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            page = browser.new_page()
            try:
                self._log(f" {self.site_config['url']}")
                page.goto(self.site_config["url"], wait_until="domcontentloaded",
                          timeout=30000)
                page.wait_for_timeout(3000)
                self._log(" Popups...")
                _dismiss_popups(page, self._log)
                for step in plan.get("search_steps", []):
                    action   = step.get("action")
                    selector = step.get("selector", "")
                    value    = step.get("value", "")
                    self._log(f"  -> {step.get('description', action)}")
                    try:
                        if action == "click":
                            page.locator(selector).first.click(timeout=5000)
                        elif action == "type":
                            for sel in [selector] + FALLBACK_INPUTS:
                                try:
                                    loc = page.locator(sel)
                                    if loc.count() > 0 and loc.first.is_visible(timeout=2000):
                                        loc.first.fill(str(value))
                                        page.wait_for_timeout(2000)
                                        page.keyboard.press("ArrowDown")
                                        page.wait_for_timeout(500)
                                        page.keyboard.press("Enter")
                                        break
                                except Exception:
                                    continue
                        elif action == "press":
                            page.keyboard.press(str(value))
                        elif action == "wait":
                            page.wait_for_timeout(int(value))
                    except Exception as e:
                        self._log(f"     {e}")
                    page.wait_for_timeout(step.get("wait_ms", 800))
                page.wait_for_timeout(4000)
                last_h = page.evaluate("document.body.scrollHeight")
                for _ in range(8):
                    page.evaluate("window.scrollBy(0,800)")
                    page.wait_for_timeout(800)
                    new_h = page.evaluate("document.body.scrollHeight")
                    if new_h == last_h: break
                    last_h = new_h
                page_text = page.inner_text("body")
                location = self.site_config.get("search_params", {}).get("location", "unknown")
                self._log(" Regex extract...")
                records = _regex_extract(page_text, location)
                if records:
                    self._log(" Ollama enrich...")
                    records = _ollama_enrich(records, page_text, location)
                else:
                    self._log("   Ollama full extract...")
                    records = _ollama_strict_extract(page_text, location)
                records = _sanity_filter(records)
                self._log(f"   {len(records)} records")
            except Exception as e:
                self._log(f" {e}")
                try:
                    page.screenshot(path="/tmp/generic_error.png")
                except Exception:
                    pass
            finally:
                browser.close()
        return records

    def run(self):
        url = self.site_config.get("url", "")
        params = self.site_config.get("search_params", {})
        location = params.get("location", "")
        date_kwargs = {
            "pickup_date": params.get("pickup_date"),
            "return_date": params.get("return_date"),
            "pickup_time": params.get("pickup_time"),
            "return_time": params.get("return_time"),
        }
        if "sixt" in url.lower():
            self._log(" SIXT dedicated flow starting...")
            records = run_sixt(location, headless=self.headless, logs=self.logs, **date_kwargs)
            site_name = "Sixt_Canada"
        elif "enterprise" in url.lower():
            self._log(" ENTERPRISE dedicated flow starting...")
            records = run_enterprise(location, headless=self.headless, logs=self.logs, **date_kwargs)
            site_name = "Enterprise_Car_Rental"
        else:
            self._log(f" Generic: {url}")
            records = self._run_generic()
            site_name = _safe(url, 24)
        return {"site": site_name, "location": location, "records": records,
                "count": len(records), "logs": self.logs}



def save_results(result: dict, output_base: str = "scraper_outputs") -> dict:
    site = result.get("site", "unknown_site")
    location = result.get("location", "unknown_location")
    folder = make_output_dir(site, location, base=output_base)
    stamp = mst_stamp()
    base_p = os.path.join(folder, stamp)
    paths = {}

    if result.get("records"):
        df = pd.DataFrame(result["records"])
        extra_cols = [c for c in df.columns if c not in CSV_COLUMNS]
        df = df[[c for c in CSV_COLUMNS if c in df.columns] + extra_cols]
        csv_path = f"{base_p}.csv"
        json_path = f"{base_p}.json"
        df.to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(result["records"], f, indent=2, ensure_ascii=False)
        paths["csv"] = csv_path
        paths["json"] = json_path
    else:
        print("\n No records — CSV/JSON skipped.")

    script_path = _save_script(site, location, folder, stamp)
    paths["script"] = script_path

    log_path = f"{base_p}.log"
    with open(log_path, "w") as f:
        f.write("\n".join(result.get("logs", [])))
    paths["logs"] = log_path

    def _display_path(path: str) -> str:
        """Relative path when possible so terminal ctrl+click opens the file."""
        try:
            return os.path.relpath(path)
        except ValueError:
            return path

    print(f"\n{'='*58}")
    print(f"  {result.get('count',0)} records  |  "
          f"{now_mst().strftime('%Y-%m-%d %H:%M:%S MST')}")
    print(f"  {_display_path(folder)}/")
    for kind, path in paths.items():
        print(f"     {kind:<8} -> {_display_path(path)}")
    print(f"{'='*58}\n")

    return paths







































































































































































