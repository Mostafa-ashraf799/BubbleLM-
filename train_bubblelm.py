#!/usr/bin/env python3
"""
الخطوة 2: تدريب BubbleLM من الصفر على train.jsonl (ناتج prepare_fineweb2.py).

بيعمل كل ده في تشغيلة واحدة:
  1. يحمّل المستندات ويقسمها: تدريب + مستندات محجوزة (held-out) للتقييم.
  2. يبني الـ bubble index.
  3. يدرّب نموذج جديد من الصفر (أوزان عشوائية) على جمل المستندات.
  4. يولّد أمثلة معايرة/اختبار تلقائيًا (أسئلة لها إجابة + أسئلة المفروض يمتنع عنها).
  5. يعاير العتبة τ ويقيّم، ويقارن ضد baselines:
       - BubbleLM كامل (LM + تحقق الفقاعات)
       - LM لوحده بعتبة ثقة
       - lookup من الـ index من غير نموذج
  6. يحفظ المشروع + report.json.

أمثلة:
  python train_bubblelm.py --data data/train.jsonl --project runs/arz --size small
  python train_bubblelm.py --config train_config.json
  python train_bubblelm.py --project runs/arz --eval-only        # إعادة التقييم فقط

الأولوية للإعدادات:  الافتراضي < --size < --config < سطر الأوامر.
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import replace

import numpy as np

try:
    import bubblelm  # لو متسطّب (pip install -e .)
except ImportError:  # أو لو السكربت جنب مجلد bubblelm-repo
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (here, os.path.join(here, "bubblelm-repo"), os.path.join(here, "..", "bubblelm-repo")):
        if os.path.isdir(os.path.join(cand, "bubblelm")):
            sys.path.insert(0, cand)
            break

from bubblelm import BubbleLM, CapacityError, EngineConfig, IndexConfig, ModelConfig, TokenizerConfig  # noqa: E402
from bubblelm import evaluate as ev  # noqa: E402
from bubblelm.model import resolve_device  # noqa: E402
from bubblelm.train import is_main  # noqa: E402

# ---------------------------------------------------------------- الإعدادات

SIZES = {
    # d_model لازم يتقسم على heads. الـ FFN دايمًا 4*d_model في الكود ده.
    "tiny":  dict(d_model=96,  layers=3,  heads=4,  maxlen=48, batch_size=128, lr=3e-3, epochs=40),
    "small": dict(d_model=256, layers=4,  heads=4,  maxlen=48, batch_size=128, lr=1.5e-3, epochs=20),
    "base":  dict(d_model=512, layers=8,  heads=8,  maxlen=48, batch_size=128, lr=6e-4, epochs=10,
                  dtype="bfloat16"),
    "large": dict(d_model=768, layers=12, heads=12, maxlen=64, batch_size=64,  lr=3e-4, epochs=6,
                  dtype="bfloat16", checkpoint=True, grad_accum=2),
    "xl":    dict(d_model=1536, layers=24, heads=16, maxlen=64, batch_size=32, lr=2e-4, epochs=4,
                  dtype="bfloat16", checkpoint=True, grad_accum=4, distributed=True, parallel="fsdp"),
}

DEFAULTS = dict(
    data="data/train.jsonl", project="runs/bubblelm", size="small", slot=None, name="",
    max_docs=0, heldout_frac=0.02, seed=0,
    max_vocab=50000, min_freq=1, rel_min_ratio=None,
    epochs=None, lr=None, batch_size=None, d_model=None, layers=None, heads=None, maxlen=None,
    dtype=None, device="auto", grad_accum=None, checkpoint=None, distributed=False, parallel=None, threads=0,
    n_pos=600, n_neg=600, profile="balanced",
    overwrite=False, allow_large=False, store_half=False, eval_only=False, skip_train=False,
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="ملف json بالإعدادات (نفس أسماء الخيارات)")
    float_keys = {"lr", "heldout_frac", "rel_min_ratio"}
    int_keys = {"epochs", "batch_size", "d_model", "layers", "heads", "maxlen", "grad_accum", "slot", "max_docs",
                "max_vocab", "min_freq", "seed", "threads", "n_pos", "n_neg"}
    for k, v in DEFAULTS.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, action="store_true", default=None)
        else:
            ap.add_argument(flag, default=None, type=float if k in float_keys else int if k in int_keys else str)
    a = ap.parse_args()
    cfg = dict(DEFAULTS)
    cli = {k: v for k, v in vars(a).items() if v is not None and k != "config"}
    file_cfg = {}
    if a.config:
        with open(a.config, encoding="utf-8") as f:
            file_cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    size = cli.get("size") or file_cfg.get("size") or cfg["size"]
    if size not in SIZES:
        sys.exit(f"--size لازم يكون واحد من: {', '.join(SIZES)}")
    cfg.update({k: v for k, v in SIZES[size].items()})
    cfg["size"] = size
    cfg.update(file_cfg)
    cfg.update(cli)
    return argparse.Namespace(**cfg)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- البيانات

def read_rows(path, max_docs):
    rows = []
    files = []
    if os.path.isdir(path):
        for root, _d, fs in sorted(os.walk(path)):
            files += [os.path.join(root, f) for f in sorted(fs) if f.endswith(".jsonl")]
    else:
        files = [path]
    for p in files:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("text"):
                    rows.append(r)
                    if max_docs and len(rows) >= max_docs:
                        return rows
    return rows


# ---------------------------------------------------------------- أمثلة المعايرة والتقييم

def make_examples(bl, heldout_rows, n_pos, n_neg, seed):
    """
    إيجابي: سؤال = الجملة من غير آخر 1-3 كلمات، والإجابة = الكلمات دي.
    سلبي (truth=None): المفروض النظام يمتنع:
       - جمل من مستندات محجوزة مش في التدريب
       - جملة متقطعة من نصفين جملتين مختلفتين
       - جملة فيها كلمة نادرة مستبدلة بكلمة عمرها ما اتشافت
    """
    rng = random.Random(seed)
    tok = bl.tokenizer
    train_sents = [tuple(s) for s in bl.corpus.lm_sentences()]
    train_set = set(train_sents)
    ok = [s for s in train_sents if 6 <= len(s) <= 40]
    if not ok:
        raise SystemExit("مفيش جمل صالحة لتوليد الأمثلة (محتاج جمل 6-40 كلمة)")
    rng.shuffle(ok)

    # --- إيجابيات: نتأكد إن الـ prefix مالوش إجابتين مختلفتين في التدريب
    cand = {}
    for s in ok[: n_pos * 4]:
        n = rng.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        cand.setdefault(s[:-n], (n, s[-n:]))
    ends = defaultdict(set)
    for s in train_sents:
        for n in (1, 2, 3):
            if len(s) > n and s[:-n] in cand:
                ends[s[:-n]].add(s[-n:])
    positives = []
    for pre, (n, ans) in cand.items():
        if len(ends[pre]) == 1:
            positives.append((" ".join(pre), " ".join(ans)))
        if len(positives) >= n_pos:
            break

    # --- سلبيات
    negatives, seen = [], set(q for q, _ in positives)

    def add_neg(tokens):
        q = " ".join(tokens)
        if len(tokens) >= 3 and q not in seen:
            seen.add(q)
            negatives.append((q, None))

    held = []
    for r in heldout_rows:
        for s in tok.sentences(r["text"]):
            t = tuple(s)
            if 6 <= len(t) <= 40 and t not in train_set:
                held.append(t)
    rng.shuffle(held)
    n_held, n_splice = int(n_neg * 0.5), int(n_neg * 0.25)
    for t in held[: n_held * 2]:
        add_neg(list(t[: -rng.choice([1, 2, 3])]))
        if len(negatives) >= n_held:
            break
    pool = ok[: max(50, n_pos)]
    for _ in range(n_splice * 5):
        if len(negatives) >= n_held + n_splice:
            break
        a, b = rng.choice(pool), rng.choice(pool)
        if a == b:
            continue
        add_neg(list(a[: len(a) // 2]) + list(b[len(b) // 2: -1]))
    idx = bl._ensure_index()
    for _ in range(n_neg * 5):
        if len(negatives) >= n_neg:
            break
        s = list(rng.choice(pool))
        cut = list(s[: -rng.choice([1, 2, 3])])
        if len(cut) < 3:
            continue
        rare = min(range(len(cut)), key=lambda i: idx.df.get(cut[i], 0))
        cut[rare] = f"zqx{rng.randrange(10 ** 6)}"
        add_neg(cut)

    exs = positives + negatives
    rng.shuffle(exs)
    calib, test = exs[0::2], exs[1::2]
    return calib, test, dict(positives=len(positives), negatives=len(negatives),
                             heldout_sentences=len(held))


def _batched(fn, prompts, size=64):
    out = []
    for i in range(0, len(prompts), size):
        out += fn(prompts[i:i + size])
    return out


def score_system(bl, examples, kind):
    prompts = [q for q, _ in examples]
    truths = [t for _, t in examples]
    if kind == "lookup":
        answers = [bl.lookup(p) for p in prompts]
    elif kind == "lm_only":
        answers = _batched(lambda ps: bl.ask_many(ps, tau=None, verify=False), prompts)
    else:
        answers = _batched(lambda ps: bl.ask_many(ps, tau=None), prompts)
    texts = [None if a.status == "abstained" else a.text for a in answers]
    scores = np.array([a.verification for a in answers], dtype=float)
    return truths, texts, scores


def eval_system(bl, calib, test, kind):
    c_truths, c_texts, c_scores = score_system(bl, calib, kind)
    unk, cor, pres = ev.arrays(c_truths, c_texts, bl._norm)
    tau = ev.tune_threshold(c_scores, cor, unk, pres)
    t_truths, t_texts, t_scores = score_system(bl, test, kind)
    unk, cor, pres = ev.arrays(t_truths, t_texts, bl._norm)
    m = ev.metrics(t_scores, cor, unk, pres, tau)
    m["recall_at_precision_99"] = ev.recall_at_precision(t_scores, cor, unk, pres, 0.99)
    m["tau"] = None if tau == float("-inf") else float(tau)
    return m


def print_table(report):
    cols = ["recall", "precision", "hallucination_rate", "coverage", "utility", "recall_at_precision_99"]
    print("\n" + "نظام".ljust(26) + "".join(c[:14].rjust(16) for c in cols))
    for name, m in report["systems"].items():
        print(name.ljust(26) + "".join(f"{m[c]:16.3f}" for c in cols))
    print()


# ---------------------------------------------------------------- التشغيل

def build_engine(a):
    mc = ModelConfig(
        arch="transformer", d_model=a.d_model, layers=a.layers, heads=a.heads, hidden=4 * a.d_model,
        maxlen=a.maxlen, epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, seed=a.seed, threads=a.threads,
        device=a.device, dtype=a.dtype or "float32", grad_accum=a.grad_accum or 1,
        checkpoint=bool(a.checkpoint), distributed=bool(a.distributed),
        parallel=a.parallel or "ddp")
    ic = IndexConfig() if a.rel_min_ratio is None else IndexConfig(rel_min_ratio=a.rel_min_ratio)
    cfg = EngineConfig(tokenizer=TokenizerConfig(min_freq=a.min_freq, max_vocab=a.max_vocab), index=ic, model=mc,
                       abstain_text="مش متأكد.")
    return BubbleLM(a.project, cfg)


def main():
    a = parse_args()
    if a.d_model % a.heads:
        sys.exit(f"d_model ({a.d_model}) لازم يتقسم على heads ({a.heads})")
    os.makedirs(a.project, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)

    log(f"مشروع: {a.project} | حجم: {a.size} | جهاز: {resolve_device(a.device)}")
    heldout = []
    if a.eval_only:
        bl = BubbleLM(a.project)
        if not len(bl.corpus):
            sys.exit("المشروع فاضي: شغّل التدريب الأول")
        rows = read_rows(a.data, a.max_docs) if os.path.exists(a.data) else []
        train_ids = {d.doc_id for d in bl.corpus.docs}
        heldout = [r for r in rows if str(r.get("id")) not in train_ids][:2000]
    else:
        if not os.path.exists(a.data):
            sys.exit(f"مش لاقي {a.data}. شغّل prepare_fineweb2.py الأول")
        rows = read_rows(a.data, a.max_docs)
        if len(rows) < 50:
            sys.exit(f"البيانات قليلة ({len(rows)} مستند)")
        random.Random(a.seed).shuffle(rows)
        n_held = max(10, int(len(rows) * a.heldout_frac)) if a.heldout_frac > 0 else 0
        heldout, train_rows = rows[:n_held], rows[n_held:]
        log(f"مستندات: تدريب={len(train_rows)} محجوز={len(heldout)}")
        bl = build_engine(a)
        t0 = time.time()
        for r in train_rows:
            bl.add_text(r["text"], str(r.get("id", "")) or None)
        bl.build()
        log(f"الـ index اتبنى في {time.time() - t0:.0f}ث: {json.dumps(bl.stats())}")

        if not a.skip_train:
            mc = bl.config.model
            from bubblelm.presets import exact_params
            from bubblelm.tokenizer import Vocab
            vsz = len(Vocab.build(list(bl.corpus.lm_sentences()), a.min_freq, a.max_vocab))
            log(f"vocab={vsz} | تقريبًا {exact_params(mc, vsz) / 1e6:.1f}M باراميتر | "
                f"جمل التدريب={sum(1 for _ in bl.corpus.lm_sentences())}")
            t0 = time.time()
            try:
                slot = bl.train(slot=a.slot, name=a.name or a.size, overwrite=a.overwrite,
                                allow_large=a.allow_large, log=log, store_half=a.store_half)
            except CapacityError as e:
                sys.exit(f"{e}\nصغّر --size أو استخدم --allow-large / --grad-accum / --checkpoint")
            log(f"التدريب خلص في {time.time() - t0:.0f}ث (slot {slot})")
            if not is_main():
                return
            bl.save()

    if a.profile:
        bl.set_profile(a.profile)

    log("بولّد أمثلة المعايرة والتقييم...")
    calib, test, info = make_examples(bl, heldout, a.n_pos, a.n_neg, a.seed)
    log(f"أمثلة: معايرة={len(calib)} اختبار={len(test)} | {info}")
    tau = bl.calibrate(calib)
    log(f"τ المحفوظة للـ slot = {tau}")

    systems = {}
    for label, kind in (("BubbleLM (LM+verify)", "bubble"), ("LM only (confidence)", "lm_only"),
                        ("index lookup (no LM)", "lookup")):
        t0 = time.time()
        systems[label] = eval_system(bl, calib, test, kind)
        log(f"قيّمت {label} في {time.time() - t0:.0f}ث")

    report = dict(config=vars(a), stats=bl.stats(), examples=info, tau=tau, systems=systems,
                  versions=bl.versions())
    print_table(report)
    with open(os.path.join(a.project, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(a.project, "eval_examples.jsonl"), "w", encoding="utf-8") as f:
        for split, exs in (("calib", calib), ("test", test)):
            for q, t in exs:
                f.write(json.dumps(dict(split=split, prompt=q, truth=t), ensure_ascii=False) + "\n")
    bl.save()
    log(f"خلص. التقرير: {os.path.join(a.project, 'report.json')}")


if __name__ == "__main__":
    main()
