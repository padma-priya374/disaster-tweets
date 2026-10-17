import json, math, os, re
import azure.functions as func

with open(os.path.join(os.path.dirname(__file__), "..", "model.json")) as f:
    MODEL = json.load(f)
INV = {i: w for w, i in MODEL["vocab"].items()}


def clean(t):
    t = re.sub(r"http\S+|@\w+", "", t.lower())
    return " ".join(re.findall(r"[a-z']+", t))


def predict(text):
    counts = {}
    for w in clean(text).split():
        if w in MODEL["vocab"]:
            i = MODEL["vocab"][w]
            counts[i] = counts.get(i, 0) + 1
    vec = {i: c * MODEL["idf"][i] for i, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1
    contrib = {i: (v / norm) * MODEL["coef"][i] for i, v in vec.items()}
    p = 1 / (1 + math.exp(-(sum(contrib.values()) + MODEL["intercept"])))
    top = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    if p >= MODEL["high"]:
        label = "Real disaster"
    elif p <= MODEL["low"]:
        label = "Not a disaster"
    else:
        label = "Uncertain - needs human review"
    return {
        "label": label,
        "probability": round(p, 3),
        "words": [{"word": INV[i], "effect": round(c, 3)} for i, c in top],
    }


def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        text = req.get_json().get("text", "")
    except ValueError:
        return func.HttpResponse("Send JSON like {\"text\": \"...\"}", status_code=400)
    return func.HttpResponse(
        json.dumps(predict(text)), mimetype="application/json"
    )
