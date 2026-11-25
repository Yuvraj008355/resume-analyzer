import json
import sqlite3

DB_PATH = "resume_analyzer.db"

JOBS = [
    ("Python Backend Developer", "TechNova",
     "Build REST APIs with Flask and Django, work with SQL databases, use git and write clean OOP code.",
     "python,flask,django,sql,rest api,oop,git"),
    ("Frontend Developer", "PixelWorks",
     "Create responsive interfaces using React, JavaScript, HTML, CSS and Bootstrap.",
     "html,css,javascript,react,bootstrap,git"),
    ("Full Stack Developer", "StackLabs",
     "Develop web apps end to end with React, Node.js, MongoDB and SQL, exposing REST APIs.",
     "html,css,javascript,react,node.js,mongodb,sql,rest api"),
    ("Data Analyst", "InsightIQ",
     "Analyze data with Python, SQL, pandas and Excel, and build dashboards in Power BI or Tableau.",
     "python,sql,excel,pandas,numpy,data analysis,power bi,tableau"),
    ("Machine Learning Engineer", "DeepMind Labs",
     "Train and deploy machine learning and deep learning models with scikit-learn, TensorFlow and PyTorch.",
     "python,machine learning,deep learning,scikit-learn,tensorflow,pytorch,numpy,pandas"),
    ("Data Scientist", "Quantify",
     "Use statistics, machine learning and SQL to extract insights from data with pandas and numpy.",
     "python,data science,machine learning,sql,pandas,numpy,statistics"),
    ("NLP Engineer", "LinguaAI",
     "Build natural language processing pipelines, text classification and chatbots using Python and deep learning.",
     "python,nlp,deep learning,pytorch,tensorflow,machine learning"),
    ("DevOps Engineer", "CloudNine",
     "Manage cloud infrastructure on AWS, containerize apps with Docker, work in Linux and git workflows.",
     "aws,docker,linux,git,github,cloud computing,python"),
    ("Java Developer", "EnterpriseSoft",
     "Develop enterprise applications in Java using OOP, data structures, algorithms and SQL.",
     "java,oop,data structures,algorithms,sql,mysql,git"),
    ("Software Engineer Intern", "StartupHub",
     "Solve problems with data structures and algorithms, write code in Python or C++, collaborate using git.",
     "python,c++,data structures,algorithms,git,problem solving,teamwork"),
    ("Administrative Manager", "OfficeWorks",
     "Manage executive calendars, schedule meetings, coordinate travel, organize company events, handle correspondence and reports, and support onboarding with strong communication and time management.",
     "communication,time management,problem solving,negotiation,scheduling,microsoft office,leadership"),
]


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT, company TEXT, description TEXT, skills TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT, ats_score REAL, top_job TEXT, skills TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        if c.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0:
            c.executemany(
                "INSERT INTO jobs (title, company, description, skills) VALUES (?,?,?,?)",
                JOBS)


def get_jobs():
    with _conn() as c:
        rows = c.execute("SELECT * FROM jobs").fetchall()
    return [{"id": r["id"], "title": r["title"], "company": r["company"],
             "description": r["description"], "skills": r["skills"].split(",")}
            for r in rows]


def save_analysis(filename, ats_score, top_job, skills):
    with _conn() as c:
        c.execute(
            "INSERT INTO analyses (filename, ats_score, top_job, skills) VALUES (?,?,?,?)",
            (filename, ats_score, top_job, json.dumps(skills)))


def get_history(limit=20):
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{"id": r["id"], "filename": r["filename"], "ats_score": r["ats_score"],
             "top_job": r["top_job"], "skills": json.loads(r["skills"]),
             "created_at": r["created_at"]} for r in rows]