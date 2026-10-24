"""Query routing + decomposition. One small LLM call returns JSON; if the call or the parsing fails, a
rule-based fallback keeps the app working."""
import json
import re

ROUTER_PROMPT = """You route questions for a Recipe & Nutrition assistant that has two documents:
- "recipe": a book of probiotic/fermented Indian recipes (ingredients, method, per-serving calories/protein/carbs/fat)
- "nutrition": the ICMR-NIN Dietary Guidelines for Indians (food groups, nutrient needs, healthy eating guidance)

Return ONLY a JSON object with keys:
  "type": "simple" (one fact), "multi" (needs several facts) or "comparison" (compares items)
  "source": "recipe", "nutrition" or "both"
  "sub_questions": list of 1-3 standalone questions (the original question itself if type is simple)

Question: {question}
JSON:"""

DEFAULT = {"type": "simple", "source": "both"}


def parse_route(text: str, question: str) -> dict:
    """Pull the first JSON object out of the model reply; validate; fall back to the heuristic on any problem."""
    try:
        m = re.search(r"\{.*\}", text, re.S)
        data = json.loads(m.group(0))
        typ = data.get("type") if data.get("type") in ("simple", "multi", "comparison") else "simple"
        src = data.get("source") if data.get("source") in ("recipe", "nutrition", "both") else "both"
        subs = [s.strip() for s in data.get("sub_questions", []) if isinstance(s, str) and s.strip()][:3]
        if not subs:
            subs = [question]
        return {"type": typ, "source": src, "sub_questions": subs, "via": "llm"}
    except Exception:
        return heuristic_route(question)


def heuristic_route(question: str) -> dict:
    q = question.lower()
    parts = [p.strip() for p in re.split(r"\band\b|;|\?\s+", question) if len(p.strip()) > 12] if " and " in q else []
    typ = "comparison" if re.search(r"\b(compare|versus|vs\.?|difference|which has more|higher|lower)\b", q) else "simple"
    if typ == "simple" and len(parts) >= 2:
        typ = "multi"
    recipe_words = r"\b(recipe|ingredients?|prepare|make|cook|serving|calories|kcal|energy|protein|ferment)"
    guide_words = r"\b(guideline|recommend|should i|daily|per day|intake|requirement|pregnan|children|elderly|sugar|salt|fat)"
    r, g = bool(re.search(recipe_words, q)), bool(re.search(guide_words, q))
    src = "recipe" if r and not g else "nutrition" if g and not r else "both"
    subs = parts[:3] if typ in ("multi", "comparison") and len(parts) >= 2 else [question]
    return {"type": typ, "source": src, "sub_questions": subs, "via": "heuristic"}


def route_question(question: str, llm=None) -> dict:
    if llm is None:
        return heuristic_route(question)
    try:
        reply = llm.invoke(ROUTER_PROMPT.format(question=question))
        text = getattr(reply, "content", str(reply))
    except Exception:
        return heuristic_route(question)
    return parse_route(text, question)
