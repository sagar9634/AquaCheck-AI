"""
chatbot.py
----------
"AquaBot" - the AI assistant inside AquaCheckAI.

It works in two modes:
  1. OFFLINE (default): a knowledge base + keyword matching + awareness of the
     user's latest water test. No internet or API key needed.
  2. LLM mode (optional): if you set the ANTHROPIC_API_KEY environment variable,
     answers come from a large language model (like ChatGPT-style chat), and
     AquaBot gives it your latest test result as context.
"""
import json
import os
import re
import urllib.request

DISCLAIMER = ("\n\n_AquaCheckAI is an educational ML project - it is not a "
              "substitute for a certified laboratory test._")

# ---- Knowledge base: (keywords, answer) -------------------------------------
KNOWLEDGE = [
    (["ph", "acidic", "basic", "alkaline"],
     "**pH** measures how acidic or basic water is (0-14). WHO suggests drinking water "
     "should be about **6.5 to 8.5**. Very low pH can corrode pipes and leach metals; "
     "very high pH gives a bitter taste and reduces chlorine's effectiveness."),
    (["hardness", "calcium", "magnesium", "limescale"],
     "**Hardness** comes from dissolved calcium and magnesium. It is mostly a comfort issue "
     "(scale in kettles, dry skin) rather than a direct health risk. Under ~150 mg/L is soft-to-moderate; "
     "above ~300 mg/L is considered very hard. A water softener or filter helps."),
    (["solids", "tds", "dissolved"],
     "**Total Dissolved Solids (TDS)** are minerals, salts and metals dissolved in water. "
     "High TDS usually tastes salty or bitter and can hide harmful contaminants. "
     "Reverse osmosis (RO) filters are the common way to reduce it."),
    (["chloramine", "chlorine", "disinfectant"],
     "**Chloramines** are disinfectants used by public water suppliers to kill germs. "
     "Some is good - it keeps water safe in the pipes - but high levels can cause irritation and an unpleasant taste. "
     "Activated-carbon filters help reduce it."),
    (["sulfate", "sulphate"],
     "**Sulfate** occurs naturally in minerals. Very high levels give a bitter taste and may have a laxative effect. "
     "Around 250 mg/L is the usual taste threshold."),
    (["conductivity", "conductive"],
     "**Conductivity** shows how well water carries electricity, which depends on dissolved salts. "
     "It is a quick indicator of TDS: higher conductivity usually means more dissolved minerals."),
    (["organic", "carbon", "toc"],
     "**Organic carbon** comes from decaying natural matter. It is harmless on its own, but when "
     "water is chlorinated it reacts to form trihalomethanes - so high organic carbon often means higher THM risk."),
    (["trihalomethane", "thm", "chloroform", "byproduct", "by-product"],
     "**Trihalomethanes (THMs)** form when chlorine reacts with organic matter. Long-term exposure to high levels "
     "is linked to health risks; the US EPA limit for total THMs is **80 μg/L**. "
     "Carbon filters and boiling (with ventilation) can reduce them."),
    (["turbidity", "cloudy", "cloudiness", "murky"],
     "**Turbidity** is cloudiness caused by tiny suspended particles. Cloudy water can shelter bacteria, "
     "so lower is better - WHO recommends below **5 NTU**, ideally under 1 NTU. Filtration and settling help."),
    (["boil", "boiling", "purify", "purification", "clean", "improve", "treat", "filter"],
     "Ways to improve water quality:\n- **Boil** for 1 minute to kill germs (it does not remove chemicals or salts)\n"
     "- **Activated carbon filter** for chlorine, THMs and bad taste\n- **Reverse osmosis** for dissolved solids and many chemicals\n"
     "- **Sediment filter / settling** for turbidity\n- Always **test again** after treatment."),
    (["accuracy", "accurate", "precision", "recall", "f1", "reliable", "trust"],
     None),  # filled dynamically from meta.json
    (["model", "algorithm", "random forest", "machine learning", "ml", "train", "trained"],
     "AquaCheckAI uses a **scikit-learn Random Forest** (compared against Logistic Regression) trained from scratch on "
     "9 water-quality measurements. A Random Forest is many decision trees voting together, which handles messy, "
     "non-linear data well."),
    (["dataset", "data", "kaggle"],
     "The model learns from a water-potability table with 9 chemical properties per sample and a label "
     "(1 = safe to drink, 0 = not safe). Missing values are filled with the median of each column."),
    (["who are you", "what are you", "aquabot", "about", "yourself"],
     "I'm **AquaBot**, the assistant built into AquaCheckAI. I can explain your results, define water-quality terms, "
     "and suggest ways to improve water."),
    (["hello", "hi", "hey", "namaste", "good morning", "good evening"],
     "Hello! 👋 I'm AquaBot. Run a water check on the left, then ask me things like "
     "_\"Why is my water unsafe?\"_ or _\"What is turbidity?\"_."),
    (["thank", "thanks"],
     "You're welcome! 💧 Ask me anything else about your water."),
]

EXPLAIN_WORDS = ["why", "result", "explain", "my water", "my sample", "unsafe", "safe", "potable",
                 "drink", "problem", "wrong", "worst", "fix", "reason"]

SUGGESTIONS = ["Why is my water unsafe?", "What is turbidity?", "How accurate is the model?",
               "How can I improve water quality?"]


def _words(text):
    return set(re.findall(r"[a-z0-9\-]+", text.lower()))


def _load_meta():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "meta.json")
    with open(path) as fh:
        return json.load(fh)


def _explain_result(context):
    """Turn the latest prediction into a plain-English explanation."""
    prob = context["probability"]
    verdict = "likely **safe to drink**" if context["potable"] else "likely **NOT safe to drink**"
    lines = [f"Your latest sample is {verdict} - the model gives it a **{prob:.0f}%** chance of being potable."]

    drivers = context.get("contributions", [])
    negatives = [c for c in drivers if c["impact"] < -0.5][:3]
    positives = [c for c in drivers if c["impact"] > 0.5][:2]
    if negatives:
        lines.append("What pulled the score **down** most:")
        for c in negatives:
            amount = (str(c["value"]) + " " + c["unit"]).strip()
            lines.append(f"- **{c['label']}** ({amount}): {c['impact']:.1f} points")
    if positives:
        lines.append("What helped:")
        lines += [f"- **{c['label']}**: +{c['impact']:.1f} points" for c in positives]
    outside = [s["label"] for s in context.get("status", []) if s["state"] != "ok"]
    if outside:
        lines.append("Outside the typical safe range: " + ", ".join(outside) + ".")
    if not context["potable"]:
        lines.append("Ask me _\"How can I improve water quality?\"_ for treatment ideas.")
    return "\n".join(lines)


def _offline_reply(message, context):
    meta = _load_meta()
    words = _words(message)
    text = message.lower()

    # 1. Questions about the user's own result ("why is my water unsafe?")
    is_definition = bool(re.search(r"\b(what is|what are|what's|define|meaning of)\b", text))
    asks_about_result = bool(words & set(EXPLAIN_WORDS)) and not is_definition
    if asks_about_result:
        if context:
            return _explain_result(context) + DISCLAIMER
        if {"why", "my", "result"} & words:
            return "Run a check first (set the sliders and press **Check Water**) and I'll explain the result for you."

    # 2. Most important feature
    if "most important" in text or ("important" in words and "feature" in text):
        top = sorted(meta["importance"].items(), key=lambda kv: -kv[1])[:3]
        return "The features the model relies on most: " + ", ".join(f"**{k}** ({v*100:.0f}%)" for k, v in top) + "."

    # 3. Knowledge base: pick the entry with the most keyword hits
    best, best_score = None, 0
    for kws, ans in KNOWLEDGE:
        score = sum(2 if " " in k and k in text else 1 for k in kws if (k in words or (" " in k and k in text)))
        if score > best_score:
            best, best_score = (kws, ans), score
    if best:
        kws, ans = best
        if ans is None:
            m = meta["metrics"]
            ans = (f"On held-out test data the **{meta['best_model']}** reached **{m['accuracy']*100:.0f}% accuracy**, "
                   f"precision {m['precision']*100:.0f}%, recall {m['recall']*100:.0f}%, F1 {m['f1']*100:.0f}%. "
                   "Water potability is hard to predict from 9 numbers, so treat results as a screening hint, not a verdict.")
        return ans + (DISCLAIMER if best_score >= 1 and "hello" not in kws else "")

    return ("I'm not sure about that one. I can explain your result, define terms like pH, turbidity or THMs, "
            "describe the model, or suggest water-treatment options. Try one of the suggestions below! 💧")


def _llm_reply(message, history, context):
    """Optional: ask a hosted LLM. Only used when ANTHROPIC_API_KEY is set."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    system = ("You are AquaBot, a friendly assistant inside AquaCheckAI, a student machine-learning project "
              "that predicts whether water is potable from 9 chemical measurements. Be concise, accurate, and "
              "remind users the tool is educational and not a lab test when giving safety advice.")
    if context:
        system += "\n\nThe user's latest check (JSON): " + json.dumps(context)
    msgs = [{"role": h["role"], "content": h["content"]} for h in history[-6:]
            if h.get("role") in ("user", "assistant")]
    msgs.append({"role": "user", "content": message})
    body = json.dumps({"model": os.environ.get("AQUABOT_MODEL", "claude-sonnet-5-5"),
                       "max_tokens": 500, "system": system, "messages": msgs}).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())["content"][0]["text"]
    except Exception as exc:  # network down, bad key, etc. -> fall back to offline mode
        print("LLM call failed, using offline mode:", exc)
        return None


def reply(message, history=None, context=None):
    message = (message or "").strip()
    if not message:
        return {"reply": "Type a question and I'll do my best to help!", "mode": "offline", "suggestions": SUGGESTIONS}
    llm = _llm_reply(message, history or [], context)
    if llm:
        return {"reply": llm, "mode": "llm", "suggestions": []}
    return {"reply": _offline_reply(message, context), "mode": "offline", "suggestions": SUGGESTIONS}
