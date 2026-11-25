import io
import os
import re

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import pymupdf as fitz
except ImportError:
    import fitz

import database as db
from ml_utils import clean_text

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB
db.init_db()

# OCR engine for scanned PDFs and images (loaded only when needed)
OCR_ENGINE = None


def get_ocr_engine():
    """Load the OCR model only when a scanned file arrives (saves memory)."""
    global OCR_ENGINE
    if OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            OCR_ENGINE = RapidOCR()
        except Exception:
            return None
    return OCR_ENGINE


# Optional AI model: semantic similarity using sentence embeddings
try:
    from sentence_transformers import SentenceTransformer, util
    EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    EMBED_MODEL = None

# Trained resume category classifier (created by train_model.py)
try:
    CATEGORY_MODEL = joblib.load("resume_model.joblib")
except Exception:
    CATEGORY_MODEL = None


def predict_category(text):
    """Top 3 predicted career categories with confidence, or None if no model."""
    if CATEGORY_MODEL is None:
        return None
    probs = CATEGORY_MODEL.predict_proba([clean_text(text)])[0]
    top = probs.argsort()[::-1][:3]
    return [{"category": str(CATEGORY_MODEL.classes_[i]),
             "confidence": round(float(probs[i]) * 100, 1)} for i in top]


SKILLS = [
    "python", "java", "c++", "c", "javascript", "html", "css", "react",
    "node.js", "flask", "django", "sql", "mysql", "mongodb",
    "machine learning", "deep learning", "artificial intelligence",
    "data science", "data analysis", "pandas", "numpy", "scikit-learn",
    "tensorflow", "pytorch", "nlp", "git", "github", "docker", "aws",
    "cloud computing", "linux", "excel", "power bi", "tableau",
    "communication", "problem solving", "leadership", "teamwork",
    "rest api", "bootstrap", "oop", "data structures", "algorithms",
    "statistics",
    "negotiation", "time management", "scheduling", "calendar management",
    "microsoft office", "customer service", "market analysis",
    "relationship management", "client acquisition", "b2b sales",
    "event planning", "report writing",
]

# alias -> canonical skill name
SYNONYMS = {
    "natural language processing": "nlp",
    "communication skills": "communication",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "reactjs": "react",
    "react.js": "react",
    "nodejs": "node.js",
    "js": "javascript",
    "sklearn": "scikit-learn",
    "restful api": "rest api",
    "restful apis": "rest api",
    "apis": "rest api",
    "api": "rest api",
    "object oriented programming": "oop",
    "object-oriented programming": "oop",
    "dsa": "data structures",
    "powerbi": "power bi",
    "problem-solving": "problem solving",
    "team work": "teamwork",
}

ALLOWED_EXT = (".pdf", ".png", ".jpg", ".jpeg")


def allowed_file(filename):
    return filename.lower().endswith(ALLOWED_EXT)


def detect_type(head):
    """Check the real file type from the first bytes, not the extension."""
    if head.startswith(b"%PDF-"):
        return "pdf"
    if head.startswith(b"\x89PNG"):
        return "png"
    if head.startswith(b"\xff\xd8"):
        return "jpg"
    return None


def extract_text_from_pdf(data):
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise PdfReadError("PDF is password protected")
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def ocr_document(data, filetype, max_pages=3):
    """Read text from a scanned PDF or an image using OCR."""
    engine = get_ocr_engine()
    if engine is None:
        raise RuntimeError("OCR not installed")
    doc = fitz.open(stream=data, filetype=filetype)
    lines = []
    for page in list(doc)[:max_pages]:
        zoom = min(2, 2000 / max(page.rect.width, page.rect.height))
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        result, _ = engine(img)
        if result:
            lines.extend(item[1] for item in result)
    return "\n".join(lines)


def _pattern(term):
    # letters, digits, + and # around a term mean it is part of another word
    return r"(?<![\w+#])" + re.escape(term) + r"(?![\w+#])"


def extract_skills(text):
    # one space between words, so skills split across lines are still found
    text = re.sub(r"\s+", " ", text.lower())
    found = set()

    for skill in SKILLS:
        if skill == "c":
            # a bare "c" is ambiguous, so only count it in a programming context
            if re.search(r"(?<![\w+#])c(?![\w+#])\s*(,|/|programming|language)", text):
                found.add("c")
            continue
        if re.search(_pattern(skill), text):
            found.add(skill)

    for alias, canonical in SYNONYMS.items():
        if re.search(_pattern(alias), text):
            found.add(canonical)

    return sorted(found)


def tfidf_similarity(a, b):
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        m = vec.fit_transform([a.lower(), b.lower()])
        return float(cosine_similarity(m[0:1], m[1:2])[0][0])
    except ValueError:
        return 0.0


def semantic_similarity(a, b):
    """Returns (score from 0 to 100, name of the method used)."""
    if EMBED_MODEL is not None:
        emb = EMBED_MODEL.encode([a[:5000], b[:5000]], convert_to_tensor=True)
        sim = float(util.cos_sim(emb[0], emb[1]))
        # related documents usually score 0.3 to 0.7, so stretch to 0 to 100
        scaled = max(0.0, min(1.0, (sim - 0.15) / 0.55))
        return round(scaled * 100, 2), "Semantic AI (sentence embeddings)"
    return round(tfidf_similarity(a, b) * 100, 2), "TF-IDF keyword overlap"


def combined_score(skill_pct, text_pct):
    return round(max(0, min(100, 0.6 * skill_pct + 0.4 * text_pct)), 2)


def generate_suggestions(text, resume_skills, missing_skills):
    low = text.lower()
    tips = []

    if len(text.split()) < 150:
        tips.append("Your resume is quite short. Add more detail about your work.")
    if "project" not in low:
        tips.append("Add a Projects section describing what you built and the tech used.")
    if not any(w in low for w in ("education", "b.tech", "btech", "bachelor", "degree")):
        tips.append("Include your education details.")
    if not any(w in low for w in ("experience", "internship", "intern")):
        tips.append("Add internships or experience. Even small ones count.")
    if not re.search(r"\d+\s*(%|\+|x|users|projects|students)", low):
        tips.append("Add measurable results (for example, 'improved accuracy by 12%').")
    if not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text):
        tips.append("No email address detected. Make sure contact details are present.")
    if "github" not in low and "linkedin" not in low:
        tips.append("Add your GitHub and/or LinkedIn links.")
    if not resume_skills:
        tips.append("Add a clear Skills section with relevant technical skills.")
    if missing_skills:
        tips.append("Skills in the job description you don't mention: "
                    + ", ".join(missing_skills[:8]))
    if not tips:
        tips.append("Good structure. Keep adding measurable achievements.")
    return tips


@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    f = request.files.get("resume")
    jd = request.form.get("job_description", "").strip()

    if not f or not f.filename or not allowed_file(f.filename):
        return jsonify(error="Please upload a PDF, PNG or JPG resume."), 400

    data = f.read()
    ftype = detect_type(data[:8])
    if ftype is None:
        return jsonify(error="That file is not a valid PDF or image."), 400

    ocr_used = False
    try:
        if ftype == "pdf":
            text = extract_text_from_pdf(data)
            if len(text.strip()) < 50:      # probably a scanned PDF
                text = ocr_document(data, "pdf")
                ocr_used = True
        else:
            text = ocr_document(data, ftype)
            ocr_used = True
    except RuntimeError:
        return jsonify(error="OCR is not installed. Run: pip install pymupdf rapidocr-onnxruntime"), 503
    except Exception:
        app.logger.exception("File read failed")
        return jsonify(error="Unable to read this file (corrupt or password protected)."), 422

    if not text.strip():
        return jsonify(error="No text could be read. Try a clearer, higher quality file."), 422

    skills = extract_skills(text)

    # ATS compatibility against the pasted job description (optional)
    ats = None
    if jd:
        jd_skills = extract_skills(jd)
        matched = [s for s in jd_skills if s in skills]
        missing = [s for s in jd_skills if s not in skills]
        text_pct, method = semantic_similarity(text, jd)
        skill_pct = len(matched) / len(jd_skills) * 100 if jd_skills else None
        score = combined_score(skill_pct, text_pct) if jd_skills else text_pct
        ats = {
            "score": score,
            "text_score": text_pct,
            "method": method,
            "skill_score": round(skill_pct, 2) if skill_pct is not None else None,
            "matched": matched,
            "missing": missing,
        }

    # Job matching against the SQL jobs table
    matches = []
    for job in db.get_jobs():
        m = [s for s in job["skills"] if s in skills]
        miss = [s for s in job["skills"] if s not in skills]
        sem, _ = semantic_similarity(text, job["description"])
        skill_pct = len(m) / len(job["skills"]) * 100
        matches.append({
            "title": job["title"],
            "company": job["company"],
            "score": combined_score(skill_pct, sem),
            "matched": m,
            "missing": miss,
        })
    matches.sort(key=lambda j: j["score"], reverse=True)
    matches = matches[:5]

    suggestions = generate_suggestions(text, skills, ats["missing"] if ats else [])

    overall = ats["score"] if ats else matches[0]["score"]
    db.save_analysis(f.filename, overall, matches[0]["title"], skills)

    return jsonify(
        filename=f.filename,
        word_count=len(text.split()),
        skills=skills,
        ats=ats,
        jobs=matches,
        suggestions=suggestions,
        ocr_used=ocr_used,
        categories=predict_category(text),
    )


@app.route("/api/history")
def api_history():
    return jsonify(db.get_history())


@app.errorhandler(413)
def file_too_large(error):
    return jsonify(error="File size exceeds 5 MB. Please upload a smaller file."), 413


@app.errorhandler(500)
def server_error(error):
    return jsonify(error="Something went wrong on the server."), 500


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=5000)