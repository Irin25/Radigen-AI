import os
# Always run from this file's folder, so thresholds.json, checkpoints/ and
# faiss_index/ are found no matter where the server is started from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from inference import Predictor
from retrieve import retrieve_for_class
from chatbot import reply as chat_reply
from extras import build_report, save_and_compare, status as engine_status, history, generate_report

# The built website (made by `npm run build` in the frontend folder)
DIST_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend", "dist"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024      # 16 MB upload cap
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])  # Svelte dev server

predictor = Predictor()          # model is loaded once, when the server starts


def _ratio(c):
    """How far a class score is above its own threshold (1.0 = exactly at the threshold)."""
    t = c.get("threshold") or 0
    return c["probability"] / t if t > 0 else 0.0


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
def analyze():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="No file sent (form field must be named 'image')"), 400
    try:
        result = predictor.predict(f.stream)        # processed in memory; the upload is never saved
    except ValueError as e:
        return jsonify(error=str(e)), 400

    classes = result["classes"]

    # Most likely finding = the flagged class furthest above its own threshold.
    # Thresholds are used here only to choose; they are not sent to the website.
    flagged = sorted((c for c in classes if c["positive"] and c["name"] != "No Finding"), key=_ratio, reverse=True)
    top, close_to = None, None
    if flagged:
        best = flagged[0]
        match = next((x for x in result["findings"] if x["name"] == best["name"]), None)
        top = {"name": best["name"], "probability": best["probability"],
               "heatmap": match["heatmap"] if match else None,
               "description": retrieve_for_class(best["name"])}
        if len(flagged) > 1 and _ratio(flagged[1]) >= 0.9 * _ratio(best):
            close_to = flagged[1]["name"]

    # The remaining classes, highest score first, so the percentages read in descending order.
    others = [c for c in classes if not top or c["name"] != top["name"]]
    ranked = [{"name": c["name"], "probability": c["probability"]}
              for c in sorted(others, key=lambda c: c["probability"], reverse=True)]

    age = request.form.get("age", type=int)
    age = age if age is not None and 0 <= age <= 120 else None
    pid = request.form.get("pid", "").strip()[:40]
    symptoms = request.form.get("symptoms", "").strip()[:200]

    return jsonify(
        top=top,
        ranked=ranked,
        close_to=close_to,
        report=build_report(top, age, symptoms, ranked, close_to),
        comparison=save_and_compare(pid, age, classes, top["name"] if top else None),
    )


@app.post("/api/chat")
def chat():
    d = request.get_json(silent=True) or {}
    q = (d.get("question") or "").strip()[:300]
    if not q:
        return jsonify(error="Ask a question first."), 400
    return jsonify(chat_reply(q, d.get("history"), d.get("scan")))


@app.get("/api/status")
def api_status():
    return jsonify(engine_status())


@app.get("/api/history")
def api_history():
    return jsonify(history(request.args.get("pid", "").strip()[:40]))


@app.post("/api/report")
def api_report():
    d = request.get_json(silent=True) or {}
    try:
        return jsonify(report=generate_report(d.get("generator"), d.get("findings", []), d.get("age"), (d.get("symptoms") or "")[:200]))
    except NotImplementedError as e:
        return jsonify(error=str(e)), 501


# ---- Serve the website itself ----
@app.get("/")
def index():
    if not os.path.isfile(os.path.join(DIST_DIR, "index.html")):
        return ("Website not built yet. In the frontend folder, run: npm run build", 404)
    return send_from_directory(DIST_DIR, "index.html")


@app.get("/<path:path>")
def website_files(path):
    if path.startswith("api/"):
        abort(404)
    if not os.path.isfile(os.path.join(DIST_DIR, path)):
        return send_from_directory(DIST_DIR, "index.html")   # client-side route such as /app
    return send_from_directory(DIST_DIR, path)


@app.errorhandler(413)
def too_big(e):
    return jsonify(error="File is too large (max 16 MB)."), 413


@app.errorhandler(Exception)
def unexpected(e):
    # Normal HTTP errors (404, 405...) keep their own code
    if isinstance(e, HTTPException):
        return jsonify(error=e.description), e.code
    # Anything else: send a readable message to the website instead of an HTML error page
    app.logger.exception("Unexpected error")
    return jsonify(error=f"Server error: {e}"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)