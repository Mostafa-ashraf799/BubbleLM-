#!/usr/bin/env python3
"""
اسأل مشروع BubbleLM اتدرّب. السؤال = بداية جملة ممكن النص يكملها.

  python ask_bubblelm.py runs/bubblelm-arz
  python ask_bubblelm.py runs/bubblelm-arz --think medium --scratchpad
  python ask_bubblelm.py runs/bubblelm-arz -q "القاهرة هي"
"""
import argparse
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
for cand in (here, os.path.join(here, "bubblelm-repo"), os.path.join(here, "..", "bubblelm-repo")):
    if os.path.isdir(os.path.join(cand, "bubblelm")):
        sys.path.insert(0, cand)
        break
from bubblelm import BubbleLM  # noqa: E402


def show(a, scratchpad):
    print(f"  -> {a.text}")
    print(f"     status={a.status} verification={a.verification:.3f} tau={a.tau} reason={a.reason}")
    for e in a.evidence[:2]:
        print(f"     دليل [{e['doc']}] {e['sentence']}  (coverage {e['coverage']})")
    if a.thinking:
        print(a.thinking_text())
    if scratchpad and a.scratchpad:
        print(a.scratchpad.render())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("-q", "--question")
    ap.add_argument("--think", choices=["low", "medium", "high"])
    ap.add_argument("--scratchpad", action="store_true")
    ap.add_argument("--natural", action="store_true", help="سؤال عادي بدل بداية جملة")
    ap.add_argument("--version", type=int)
    a = ap.parse_args()
    bl = BubbleLM(a.project)
    if a.question:
        show(bl.ask(a.question, version=a.version, think=a.think, natural=a.natural), a.scratchpad)
        return
    print("اكتب بداية جملة (Enter فاضي للخروج)")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if not q:
            break
        show(bl.ask(q, version=a.version, think=a.think, natural=a.natural), a.scratchpad)


if __name__ == "__main__":
    main()
