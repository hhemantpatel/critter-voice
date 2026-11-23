import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datetime import datetime
from tommy.durations import parse_span, parse_clock

SPAN = {
    "25 minutes": 1500, "25min": 1500, "25m": 1500, "1 hour": 3600, "an hour": 3600, "1.5 hours": 5400,
    "2h30m": 9000, "1 hour 30 minutes": 5400, "90 seconds": 90, "45 sec": 45, "half an hour": 1800,
    "a minute": 60, "an hour and a half": 5400, "2 and a half hours": 9000, "twenty five minutes": 1500,
    "one hundred twenty minutes": 7200, "thirty seconds": 30, "pachees minute": 1500, "pachis min": 1500,
    "tees minute": 1800, "ikattis minute": 1860, "das minute": 600, "paanch minute": 300, "bees minat": 1200, "aadha ghanta": 1800,
    "dedh ghanta": 5400, "dhai ghante": 9000, "sawa ghanta": 4500, "sawa do ghante": 8100,
    "paune do ghante": 6300, "sadhe teen ghante": 12600, "ek ghanta": 3600, "do ghante": 7200,
    "ek ghanta pachees minute": 5100, "chalis minute": 2400, "pachas minute": 3000, "sau minute": 6000,
    "in 25 minutes please": 1500, "15 min baad": 900, "1 din": 86400,
}
CLOCK_NOW = datetime(2026, 11, 22, 14, 0)
CLOCK = {   # phrase -> (hour, minute, days ahead)
    "5pm": (17, 0, 0), "5:30 pm": (17, 30, 0), "17:45": (17, 45, 0), "shaam 5 baje": (17, 0, 0),
    "shaam paune paanch baje": (16, 45, 0), "subah 7 baje": (7, 0, 1), "kal subah 7 baje": (7, 0, 1),
    "raat 11 baje": (23, 0, 0), "raat 2 baje": (2, 0, 1), "3 baje": (15, 0, 0), "saadhe paanch baje": (17, 30, 0),
    "sawa paanch baje": (17, 15, 0), "dedh baje": (1, 30, 1), "8 am": (8, 0, 1), "7 30 am": (7, 30, 1), "7.30 am": (7, 30, 1), "07.30 a.m.": (7, 30, 1), "5.15 baje": (17, 15, 0), "7 thirty a.m.": (7, 30, 1), "saam paanch baje": (17, 0, 0), "5 p.m.": (17, 0, 0), "tomorrow 9 am": (9, 0, 1),
}

def main():
    bad = 0
    for phrase, want in SPAN.items():
        got = parse_span(phrase)
        if got != want: bad += 1; print(f"SPAN  {phrase!r}: got {got}, want {want}")
    for phrase, (h, m, d) in CLOCK.items():
        got = parse_clock(phrase, CLOCK_NOW)
        want_day = CLOCK_NOW.day + d
        if not got or (got.hour, got.minute, got.day) != (h, m, want_day):
            bad += 1; print(f"CLOCK {phrase!r}: got {got}, want {h:02}:{m:02} day+{d}")
    for junk in ("hello", "", "play music"):
        if parse_span(junk) is not None: bad += 1; print(f"JUNK {junk!r} should be None")
    print(f"{len(SPAN) + len(CLOCK) + 3 - bad}/{len(SPAN) + len(CLOCK) + 3} passed")
    return bad

if __name__ == "__main__":
    raise SystemExit(main())
