import csv
import hmac
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from functools import lru_cache, wraps
from pathlib import Path

from flask import (
    Flask,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(
    os.environ.get("EMOTION_MODEL_PATH", str(BASE_DIR / "best_artemis_emotion_model"))
)
MAX_TEXT_LENGTH = 5000
AUTH_DB_PATH = Path(
    os.environ.get("ARTEMIS_AUTH_DB", str(BASE_DIR / "instance" / "users.sqlite3"))
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get(
    "SESSION_COOKIE_SECURE", ""
).lower() in {"1", "true", "yes"}
EMOTION_LABELS = [
    "amusement",
    "anger",
    "awe",
    "contentment",
    "disgust",
    "excitement",
    "fear",
    "sadness",
    "something else",
]
EMOTION_GUIDANCE = {
    "amusement": "Often associated with humor, playfulness, or finding something funny.",
    "anger": "May appear in language expressing frustration, irritation, or outrage.",
    "awe": "Can describe wonder or amazement in response to something meaningful or vast.",
    "contentment": "Often associated with ease, satisfaction, or a sense that things are okay.",
    "disgust": "May express strong aversion, revulsion, or rejection.",
    "excitement": "Can signal eager anticipation, enthusiasm, or high positive energy.",
    "fear": "May appear in language about worry, threat, apprehension, or feeling unsafe.",
    "sadness": "Often associated with loss, disappointment, loneliness, or low mood.",
    "something else": "A broad category for text that does not fit the other available labels.",
}


def get_secret_key():
    if app.config["SECRET_KEY"]:
        return app.config["SECRET_KEY"]

    secret_path = Path(app.instance_path) / "session.key"
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with secret_path.open("x", encoding="ascii") as secret_file:
            secret_file.write(secrets.token_hex(32))
    except FileExistsError:
        pass
    return secret_path.read_text(encoding="ascii").strip()


app.config["SECRET_KEY"] = get_secret_key()


def get_auth_connection():
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(AUTH_DB_PATH), timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def auth_connection():
    connection = get_auth_connection()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_auth_database():
    with auth_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def get_csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return session["csrf_token"]


def refresh_csrf_token():
    session["csrf_token"] = secrets.token_urlsafe(32)
    return session["csrf_token"]


def csrf_is_valid():
    submitted_token = request.headers.get("X-CSRF-Token")
    if submitted_token is None:
        submitted_token = request.form.get("csrf_token", "")
    session_token = session.get("csrf_token", "")
    return bool(session_token) and hmac.compare_digest(
        str(session_token), str(submitted_token)
    )


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("user_id"):
            if request.is_json:
                return jsonify(error="Authentication required."), 401
            return redirect(url_for("login"))
        g.current_username = session.get("username", "")
        return view(*args, **kwargs)

    return wrapped_view


@app.context_processor
def inject_template_context():
    return {
        "csrf_token": get_csrf_token,
        "current_username": session.get("username"),
    }


def get_project_data():
    metrics_path = BASE_DIR / "results" / "test_metrics.json"
    report_path = BASE_DIR / "results" / "classification_report.csv"
    config_path = MODEL_DIR / "config.json"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    labels = list(config.get("label2id", {}).keys()) or EMOTION_LABELS

    report = []
    if report_path.is_file():
        with report_path.open(encoding="utf-8", newline="") as report_file:
            report = [
                row
                for row in csv.DictReader(report_file)
                if row.get("precision") not in (None, "")
                and row.get("recall") not in (None, "")
                and row.get("f1-score") not in (None, "")
                and row.get("") in labels
            ]
    class_performance = [
        {
            "label": row[""],
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1-score"]),
            "support": int(float(row["support"])),
        }
        for row in report
    ]
    config_labels = config.get("label2id", {})
    emotion_guidance = [
        {
            "label": label,
            "description": EMOTION_GUIDANCE.get(
                label, "One of the model's available emotional categories."
            ),
        }
        for label in labels
    ]

    tracked_files = [
        ("Model weights", MODEL_DIR / "model.safetensors", "Trained classifier parameters"),
        ("Model configuration", config_path, "Architecture and label mapping"),
        ("Tokenizer", MODEL_DIR / "tokenizer.json", "Text preprocessing vocabulary"),
        ("Test metrics", metrics_path, "Held-out evaluation summary"),
        ("Classification report", report_path, "Per-class evaluation results"),
        ("Confusion matrix", BASE_DIR / "results" / "confusion_matrix.png", "Evaluation visualization"),
    ]
    assets = [
        {
            "name": name,
            "description": description,
            "path": path.relative_to(BASE_DIR).as_posix(),
            "exists": path.is_file(),
            "size": path.stat().st_size if path.is_file() else None,
        }
        for name, path, description in tracked_files
    ]
    available_count = sum(asset["exists"] for asset in assets)

    return {
        "metrics": metrics,
        "labels": labels,
        "report": report,
        "class_performance": class_performance,
        "emotion_guidance": emotion_guidance,
        "evaluation_samples": sum(item["support"] for item in class_performance),
        "model_details": {
            "architecture": config.get("architectures", ["DistilBERT classifier"])[0],
            "layers": config.get("n_layers", "—"),
            "attention_heads": config.get("n_heads", "—"),
            "hidden_size": config.get("dim", "—"),
            "max_tokens": config.get("max_position_embeddings", 512),
            "emotion_count": len(config_labels) or len(labels),
        },
        "assets": assets,
        "available_count": available_count,
        "model_ready": (MODEL_DIR / "config.json").is_file()
        and (MODEL_DIR / "model.safetensors").is_file(),
    }


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Model dependencies are missing. Install Flask, torch, and transformers."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model.eval()
    return torch, tokenizer, model


def classify_text(text):
    torch, tokenizer, model = load_model()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.inference_mode():
        probabilities = model(**inputs).logits.softmax(dim=-1)[0].tolist()

    id2label = model.config.id2label
    predictions = [
        {
            "label": id2label.get(index, id2label.get(str(index), str(index))),
            "score": float(score),
        }
        for index, score in enumerate(probabilities)
    ]
    predictions.sort(key=lambda item: item["score"], reverse=True)

    return {
        "label": predictions[0]["label"],
        "confidence": predictions[0]["score"],
        "top_predictions": predictions[:3],
        "probabilities": {
            item["label"]: item["score"] for item in predictions
        },
    }


initialize_auth_database()


@app.get("/")
def index():
    if session.get("user_id"):
        return redirect(url_for("home"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("home"))
    error = None
    username = ""

    if request.method == "POST":
        if not csrf_is_valid():
            error = "Your session expired. Refresh the page and try again."
        else:
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            if 1 <= len(username) <= 40 and 1 <= len(password) <= 128:
                with auth_connection() as connection:
                    user = connection.execute(
                        "SELECT id, username, password_hash FROM users "
                        "WHERE username = ? COLLATE NOCASE",
                        (username,),
                    ).fetchone()
                if user and check_password_hash(user["password_hash"], password):
                    session.clear()
                    session["user_id"] = user["id"]
                    session["username"] = user["username"]
                    session["csrf_token"] = secrets.token_urlsafe(32)
                    return redirect(url_for("home"))
            error = "The username or password is incorrect."

    return render_template("login.html", error=error, username=username)


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("home"))

    error = None
    username = ""
    if request.method == "POST":
        if not csrf_is_valid():
            error = "Your session expired. Refresh the page and try again."
        else:
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            password_confirmation = request.form.get("password_confirmation", "")

            if not 3 <= len(username) <= 40:
                error = "Username must be between 3 and 40 characters."
            elif not 12 <= len(password) <= 128:
                error = "Choose a password between 12 and 128 characters."
            elif password != password_confirmation:
                error = "The passwords do not match."
            else:
                try:
                    with auth_connection() as connection:
                        cursor = connection.execute(
                            "INSERT INTO users (username, password_hash) "
                            "VALUES (?, ?)",
                            (username, generate_password_hash(password)),
                        )
                except sqlite3.IntegrityError:
                    error = "That username is already in use. Choose another."
                else:
                    session.clear()
                    session["user_id"] = cursor.lastrowid
                    session["username"] = username
                    session["csrf_token"] = secrets.token_urlsafe(32)
                    return redirect(url_for("home"))

    return render_template("register.html", error=error, username=username)


@app.post("/logout")
@login_required
def logout():
    if not csrf_is_valid():
        return "Your session expired. Refresh the page and try again.", 400
    session.clear()
    return redirect(url_for("login"))


@app.get("/dashboard")
@login_required
def dashboard():
    return redirect(url_for("home"))


@app.get("/home")
@login_required
def home():
    return render_template("home.html", **get_project_data())


@app.get("/abstract")
@login_required
def abstract():
    return render_template("abstract.html", **get_project_data())


@app.get("/performance")
@login_required
def performance():
    return render_template("performance.html", **get_project_data())


@app.get("/charts")
@login_required
def charts():
    return render_template("charts.html", **get_project_data())


@app.get("/storage")
@login_required
def storage():
    return render_template("storage.html", **get_project_data())


@app.get("/prediction")
@login_required
def prediction_page():
    return render_template(
        "prediction.html",
        max_length=MAX_TEXT_LENGTH,
        text="",
        prediction=None,
        error=None,
    )


@app.post("/predict")
@login_required
def predict():
    if not csrf_is_valid():
        message = "Your security token was refreshed. Submit your text again."
        csrf_token = refresh_csrf_token()
        if request.is_json:
            return jsonify(error=message, csrf_token=csrf_token), 400
        text = request.form.get("text", "")
        if not isinstance(text, str):
            text = ""
        return render_template(
            "prediction.html",
            max_length=MAX_TEXT_LENGTH,
            text=text[:MAX_TEXT_LENGTH],
            prediction=None,
            error=message,
        ), 400

    is_json = request.is_json
    payload = request.get_json() if is_json else request.form
    if is_json and not isinstance(payload, dict):
        return jsonify(error="Request body must be a JSON object."), 400

    text = payload.get("text", "")
    if not isinstance(text, str):
        message = "The 'text' value must be a string."
        if is_json:
            return jsonify(error=message), 400
        return render_template(
            "prediction.html",
            max_length=MAX_TEXT_LENGTH,
            text="",
            prediction=None,
            error=message,
        ), 400

    text = text.strip()
    if not text:
        message = "Please enter some text to analyze."
        if is_json:
            return jsonify(error=message), 400
        return render_template(
            "prediction.html",
            max_length=MAX_TEXT_LENGTH,
            text="",
            prediction=None,
            error=message,
        ), 400

    if len(text) > MAX_TEXT_LENGTH:
        message = f"Text must be {MAX_TEXT_LENGTH} characters or fewer."
        if is_json:
            return jsonify(error=message), 400
        return render_template(
            "prediction.html",
            max_length=MAX_TEXT_LENGTH,
            text=text[:MAX_TEXT_LENGTH],
            prediction=None,
            error=message,
        ), 400

    try:
        prediction = classify_text(text)
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        app.logger.exception("Unable to load or run the emotion classifier")
        message = f"Unable to run the emotion classifier: {exc}"
        if is_json:
            return jsonify(error=message), 503
        return render_template(
            "prediction.html",
            max_length=MAX_TEXT_LENGTH,
            text=text,
            prediction=None,
            error=message,
        ), 503

    if is_json:
        return jsonify(prediction)
    return render_template(
        "prediction.html",
        max_length=MAX_TEXT_LENGTH,
        text=text,
        prediction=prediction,
        error=None,
    )


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        model_ready=(MODEL_DIR / "config.json").is_file()
        and (MODEL_DIR / "model.safetensors").is_file(),
    )


if __name__ == "__main__":
    app.run(
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
