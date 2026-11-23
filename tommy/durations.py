"""Turn time phrases into numbers. Deterministic on purpose: the model only copies the phrase.

parse_span("pachees minute")      -> 1500.0 seconds
parse_span("dedh ghante")         -> 5400.0
parse_clock("shaam paune paanch baje", now) -> datetime for the next 16:45
Both return None when nothing usable is found, and the caller then asks the user.
"""
import re
from datetime import datetime, timedelta

_EN = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_EN.update({"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
            "seventy": 70, "eighty": 80, "ninety": 90})

# Roman-script Hindi has no standard spelling, so each number lists the spellings people type.
_HI_SPELL = {
    1: "ek", 2: "do", 3: "teen tin", 4: "char chaar", 5: "paanch panch paach", 6: "chhe chhah che chah chhai",
    7: "saat sat", 8: "aath ath", 9: "nau", 10: "das dus", 11: "gyarah gyaarah", 12: "barah baarah",
    13: "terah", 14: "chaudah chaudha", 15: "pandrah pandra", 16: "solah solaah", 17: "satrah satra",
    18: "atharah athaarah atharaah", 19: "unnis unnees unees", 20: "bees bis", 21: "ikkis ikkees ekkis",
    22: "baais bais", 23: "teis teyis", 24: "chaubis chaubees", 25: "pachis pachees pacchis pachchis pachhis",
    26: "chhabbis chabbis", 27: "sattais satais", 28: "atthais athais", 29: "untis unattis",
    30: "tees tis", 31: "iktis ikattis ikatis ikattees", 32: "battis batis", 33: "taintis", 34: "chauntis", 35: "paintis",
    36: "chhattis chattis", 37: "saintis", 38: "adtis", 39: "untalis", 40: "chalis chaalis",
    41: "iktalis", 42: "bayalis", 43: "taintalis", 44: "chauvalis", 45: "paintalis", 46: "chhiyalis",
    47: "saintalis", 48: "adtalis", 49: "unchas", 50: "pachas pachaas pachaas", 55: "pachpan",
    60: "saath saadh sath", 75: "pachattar", 90: "nabbe", 
}
_HI = {w: n for n, ws in _HI_SPELL.items() for w in ws.split()}
_NUM = {**_EN, **_HI}
_HUNDRED = {"hundred", "sau", "so"}
_UNIT = {}
for secs, words in ((1, "s sec secs second seconds sekand"), (60, "m min mins minute minutes minat mint minit"),
                    (3600, "h hr hrs hour hours ghanta ghante ghanto ghanta"), (86400, "day days din roz")):
    for w in words.split():
        _UNIT[w] = secs
_FRAC = {"aadha": 0.5, "adha": 0.5, "aadhe": 0.5, "half": 0.5, "dedh": 1.5, "derh": 1.5, "dhai": 2.5,
         "dhaai": 2.5, "adhai": 2.5, "quarter": 0.25}
_FILLER = {"an", "a", "of", "ka", "ke", "ki", "ko", "me", "mein", "in", "after", "baad", "for", "the", "and",
           "aur", "please", "plz", "pls", "only", "hi", "bas"}
_TOKEN = re.compile(r"\d+(?::\d+)?(?:\.\d+)?|[a-z]+")


def _tokens(text):
    # "25min" / "2h30m" -> split digits from letters before tokenising
    text = re.sub(r"\b([ap])\.?m\b\.?", r"\1m", text.lower())             # "a.m." -> "am" (else the m reads as minutes)
    text = re.sub(r"(\d)([a-z])", r"\1 \2", text)
    text = re.sub(r"([a-z])(\d)", r"\1 \2", text)
    return _TOKEN.findall(text)


def _read_number(toks, i):
    """Read a number at toks[i]. Returns (value, next_index) or (None, i)."""
    if i >= len(toks):
        return None, i
    t = toks[i]
    if t[0].isdigit():
        if ":" in t:
            return None, i
        return float(t), i + 1
    total = cur = 0.0
    seen, j = False, i
    while j < len(toks):
        w = toks[j]
        if w in _HUNDRED and (seen or w != "so"):
            cur = (cur or 1) * 100; seen = True
        elif w in _NUM:
            if seen and w in _EN and _EN[w] < 10 and cur % 100 >= 10 and cur % 10 != 0:
                break                       # "twenty five" is one number; "twenty five three" is two
            cur += _NUM[w]; seen = True
        elif w == "and" and seen and cur >= 100 and j + 1 < len(toks) and toks[j + 1] in _NUM:
            pass
        else:
            break
        j += 1
    if not seen:
        return None, i
    return total + cur, j


def parse_span(text):
    toks = _tokens(text)
    total, i, found = 0.0, 0, False
    while i < len(toks):
        t = toks[i]
        # sawa N = N + 1/4 ; sadhe N = N + 1/2 ; paune N = N - 1/4   (sawa alone = 1.25)
        mod = {"sawa": 0.25, "sava": 0.25, "sadhe": 0.5, "saadhe": 0.5, "saade": 0.5, "paune": -0.25, "pone": -0.25}.get(t)
        if mod is not None:
            n, j = _read_number(toks, i + 1)
            if n is None:
                n, j = 1.0, i + 1
            val = n + mod
        elif t in _FRAC:
            val, j = _FRAC[t], i + 1
        elif t in ("a", "an") and i + 1 < len(toks) and toks[i + 1] in _UNIT:
            val, j = 1.0, i + 1
        else:
            n, j = _read_number(toks, i)
            if n is None:
                i += 1
                continue
            val = n
            if toks[j:j + 3] == ["and", "a", "half"]: val, j = val + 0.5, j + 3
            elif toks[j:j + 2] in (["and", "half"], ["aur", "aadha"]): val, j = val + 0.5, j + 2
        while j < len(toks) and toks[j] in _FILLER:
            j += 1
        if j < len(toks) and toks[j] in _UNIT:
            unit = _UNIT[toks[j]]
            total += val * unit; found = True; i = j + 1
            if toks[i:i + 3] == ["and", "a", "half"]: total, i = total + 0.5 * unit, i + 3     # "an hour and a half"
            elif toks[i:i + 2] == ["and", "half"]: total, i = total + 0.5 * unit, i + 2
        else:
            i = max(i + 1, j if j > i else i + 1)
    return total if found and total > 0 else None


_PERIOD = {"subah": "am", "morning": "am", "savere": "am", "dopahar": "pm", "afternoon": "pm", "dophar": "pm",
           "shaam": "pm", "sham": "pm", "saam": "pm", "evening": "pm", "raat": "night", "night": "night", "am": "am", "pm": "pm"}


def parse_clock(text, now=None):
    """Next occurrence of a wall-clock time like '5:30 pm', '17:45', 'shaam paune paanch baje'."""
    now = now or datetime.now()
    low = re.sub(r"\b(\d{1,2})\.(\d{2})(?=\s*(?:[ap]\.?m\b|baje))", r"\1:\2", text.lower())     # "7.30 am" is 7:30, not 7 and 30/100
    toks = _tokens(low)
    day_shift = 1 if any(t in ("kal", "tomorrow", "tmrw") for t in toks) else 0
    period = next((_PERIOD[t] for t in toks if t in _PERIOD), None)
    hour = minute = None
    m = re.search(r"\b(\d{1,2})[: ](\d{2})\b(?=\s*[ap]\.?m\b|$|\s*(?:baje|am|pm))", low) or re.search(r"\b(\d{1,2}):(\d{2})\b", low)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
    elif (dm := re.search(r"\b(\d{1,2})\s+([a-z]+)\s*(?=[ap]m\b|baje)", re.sub(r"\b([ap])\.?m\b\.?", r"\1m", low))) and _NUM.get(dm.group(2), 0) >= 10:
        hour, minute = int(dm.group(1)), int(_NUM[dm.group(2)])           # "7 thirty am"
    else:
        mod = next(({"sawa": 0.25, "sava": 0.25, "sadhe": 0.5, "saadhe": 0.5, "paune": -0.25, "pone": -0.25}[t]
                    for t in toks if t in ("sawa", "sava", "sadhe", "saadhe", "paune", "pone")), None)
        frac = next((_FRAC[t] for t in toks if t in ("dedh", "derh", "dhai", "dhaai", "adhai")), None)
        n = None
        for i in range(len(toks)):
            n, _ = _read_number(toks, i)
            if n is not None and toks[i] not in _FRAC:
                break
            n = None
        if frac is not None and n is None:
            n = frac                                              # dedh baje = 1:30, dhai baje = 2:30
        if n is None:
            return None
        if mod is not None:
            n = n + mod
        hour, minute = int(n), int(round((n - int(n)) * 60))
        if minute < 0:
            hour, minute = hour - 1, minute + 60
    if hour > 23 or minute > 59:
        return None
    if period == "pm" and hour < 12: hour += 12
    elif period == "am" and hour == 12: hour = 0
    elif period == "night":
        hour = hour + 12 if 6 <= hour <= 11 else (0 if hour == 12 else hour)   # raat 11 = 23:00, raat 2 = 02:00
    base = now.replace(hour=hour % 24, minute=minute, second=0, microsecond=0) + timedelta(days=day_shift)
    if period is None and (not m or "baje" in low) and 1 <= hour <= 12 and not day_shift:
        # bare "5 baje": whichever of 05:00 / 17:00 comes next
        cands = [base.replace(hour=hour % 12), base.replace(hour=hour % 12 + 12)]
        future = [c for c in cands if c > now]
        return min(future) if future else min(cands) + timedelta(days=1)
    return base if base > now else base + timedelta(days=1)


_NOW = {"now", "abhi", "turant", "immediately", "asap"}


def is_now(text):
    """True for 'sleep now', 'abhi sula do', 'right now' ... (an action with no delay)."""
    return bool(_NOW & set(_tokens(text)))


def human(seconds):
    s = int(round(seconds))
    h, rem = divmod(s, 3600); m, sec = divmod(rem, 60)
    return " ".join(p for p in (f"{h} h" if h else "", f"{m} min" if m else "", f"{sec} s" if sec and not h else "") if p) or "0 s"
