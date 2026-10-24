"""Run the evaluation set through the full pipeline with RAGAS scoring. Needs GROQ_API_KEY and the index.
    python scripts/run_eval.py [--pause 20]
Free-tier Groq limits tokens per minute, so questions are spaced out; expect ~1-2 minutes per answerable question."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app import config, evaluation  # noqa: E402
from backend.app.chain import RagService  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pause", type=int, default=20, help="seconds between questions")
ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
args = ap.parse_args()

qs = json.load(open(config.ROOT / "data" / "eval" / "questions.json"))
qs = qs[: args.limit] if args.limit else qs
svc, rows, ok = RagService(), [], 0
for item in qs:
    res = svc.ask(item["q"])
    rec = evaluation.evaluate_and_log(res)
    behaved = (res["refused"] == (item["expect"] == "refuse"))
    ok += behaved
    rows.append((item["q"], res["refused"], behaved, rec))
    print(f"{'OK ' if behaved else 'BAD'} refused={res['refused']!s:5} F={rec['faithfulness']} AR={rec['answer_relevancy']} "
          f"CP={rec['context_precision']} [{rec['status']}] {item['q']}")
    time.sleep(args.pause)

print(f"\nRefusal/answer behaviour correct on {ok}/{len(qs)} questions")
for m in ("faithfulness", "answer_relevancy", "context_precision"):
    v = [r[3][m] for r in rows if r[3][m] is not None]
    print(f"mean {m}: {sum(v) / len(v):.3f} over {len(v)} scored" if v else f"mean {m}: no scores")
