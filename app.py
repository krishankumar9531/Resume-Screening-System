"""
Resume Screening & Candidate Shortlisting System
================================================

A Streamlit decision-support tool that ranks resumes against a job description.

Scoring blends two independent signals:

1. JD fit      - skills, experience and education measured against the job description
2. Model score - probability from your trained scikit-learn pipeline
                 (resume_screening_model.pkl, trained on ai_resume_screening.csv)

Files expected next to this script:
    resume_screening_model.pkl   (optional - retrained from the CSV if it will not load)
    ai_resume_screening.csv      (optional - used for retraining, metrics and the playground)

Install:
    pip install streamlit pandas numpy scikit-learn joblib pymupdf python-docx

Run:
    streamlit run app.py

Tip: the interface is designed for Streamlit's light theme. To pin it, create
.streamlit/config.toml containing:
    [theme]
    base = "light"
"""

from __future__ import annotations

import hashlib
import html
import io
import math
import re
import warnings
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CONSTANTS
# ============================================================

APP_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
MODEL_PATH = APP_DIR / "resume_screening_model.pkl"
DATA_PATH = APP_DIR / "ai_resume_screening.csv"

FEATURE_COLUMNS = [
    "years_experience",
    "skills_match_score",
    "education_level",
    "project_count",
    "resume_length",
    "github_activity",
]
NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c != "education_level"]
TARGET_COLUMN = "shortlisted"

# Ranges seen in the training data. Values are clamped to these before scoring
# so that one unusual resume cannot push the model outside what it has seen.
FEATURE_LIMITS = {
    "years_experience": (0, 15),
    "skills_match_score": (0, 100),
    "project_count": (0, 25),
    "resume_length": (150, 900),
    "github_activity": (0, 842),
}
DEFAULT_GITHUB_ACTIVITY = 321      # dataset median; resumes do not contain this number
NEUTRAL_SKILL_SCORE = 74.3         # dataset median, used when a JD lists no skills

EDU_LEVELS = ["Unknown", "High School", "Bachelors", "Masters", "PhD"]
EDU_RANK = {label: i for i, label in enumerate(EDU_LEVELS)}
REQ_EDU_OPTIONS = ["Any", "High School", "Bachelors", "Masters", "PhD"]

STATUS_SHORTLISTED = "Shortlisted"
STATUS_REVIEW = "Review"
STATUS_REJECTED = "Not shortlisted"
STATUS_CLASS = {STATUS_SHORTLISTED: "ok", STATUS_REVIEW: "warn", STATUS_REJECTED: "bad"}
HR_DECISIONS = ["Auto", "Shortlist", "Reject"]

SAMPLE_JD = """Backend Developer

We are hiring a Backend Developer with 2+ years of experience building web services.

Required skills: Python, Django, REST API, PostgreSQL, Docker, AWS and Git.
Nice to have: Redis, CI/CD.

Education: B.Tech / B.E. in Computer Science, or MCA.
"""

# ============================================================
# SKILL TAXONOMY  (canonical name -> aliases)
# ============================================================

SKILL_ALIASES: dict[str, list[str]] = {
    # languages
    "python": [],
    "java": [],
    "javascript": ["js", "ecmascript"],
    "typescript": [],
    "c++": ["cpp"],
    "c": [],
    "c#": ["csharp", "c sharp"],
    "golang": ["go lang"],
    "rust": [],
    "kotlin": [],
    "swift": [],
    "php": [],
    "ruby": [],
    "scala": [],
    "matlab": [],
    "bash": ["shell scripting"],
    "sql": [],
    # web & backend
    "html": ["html5"],
    "css": ["css3"],
    "react": ["react.js", "reactjs"],
    "angular": ["angularjs"],
    "vue": ["vue.js", "vuejs"],
    "next.js": ["nextjs"],
    "node.js": ["nodejs", "node js", "node"],
    "express.js": ["expressjs"],
    "django": [],
    "flask": [],
    "fastapi": [],
    "spring boot": ["springboot"],
    "spring": [],
    "hibernate": [],
    ".net": ["dotnet", "asp.net"],
    "jquery": [],
    "tailwind": ["tailwind css"],
    "bootstrap": [],
    "rest api": ["rest apis", "restful", "restful api", "restful apis", "rest"],
    "graphql": [],
    "microservices": ["micro-services"],
    # data
    "mysql": [],
    "postgresql": ["postgres"],
    "mongodb": ["mongo"],
    "oracle": [],
    "sqlite": [],
    "redis": [],
    "elasticsearch": [],
    "cassandra": [],
    "nosql": ["no sql"],
    "snowflake": [],
    "bigquery": [],
    "hadoop": [],
    "spark": ["apache spark", "pyspark"],
    "kafka": [],
    "airflow": [],
    "etl": [],
    "data analysis": ["data analytics"],
    "statistics": ["statistical analysis"],
    # ML / AI
    "machine learning": ["ml"],
    "deep learning": [],
    "nlp": ["natural language processing"],
    "computer vision": [],
    "tensorflow": [],
    "pytorch": [],
    "keras": [],
    "scikit-learn": ["sklearn", "scikit learn"],
    "pandas": [],
    "numpy": [],
    "scipy": [],
    "matplotlib": [],
    "seaborn": [],
    "opencv": [],
    "xgboost": [],
    "hugging face": ["huggingface"],
    "llm": ["llms", "large language models"],
    "langchain": [],
    "generative ai": ["genai", "gen ai"],
    # BI
    "power bi": ["powerbi"],
    "tableau": [],
    "excel": ["ms excel", "microsoft excel", "advanced excel"],
    "looker": [],
    # cloud & devops
    "docker": [],
    "kubernetes": ["k8s"],
    "aws": ["amazon web services"],
    "azure": [],
    "gcp": ["google cloud"],
    "terraform": [],
    "jenkins": [],
    "ci/cd": ["cicd", "ci cd", "continuous integration"],
    "git": [],
    "github": [],
    "gitlab": [],
    "linux": [],
    "ansible": [],
    # mobile
    "android": [],
    "ios": [],
    "flutter": [],
    "react native": [],
    # practices
    "agile": [],
    "scrum": [],
    "jira": [],
    "unit testing": ["junit", "pytest"],
    "selenium": [],
    "postman": [],
}

ALIAS_TO_CANON: dict[str, str] = {}
for _canon, _aliases in SKILL_ALIASES.items():
    ALIAS_TO_CANON[_canon] = _canon
    for _alias in _aliases:
        ALIAS_TO_CANON[_alias] = _canon

# Extra regex appended to a skill pattern to avoid common false positives.
SKILL_GUARDS = {"excel": r"(?!\s+(?:in|at|as)\b)"}

PRETTY_SKILLS = {
    "aws": "AWS", "gcp": "GCP", "sql": "SQL", "html": "HTML", "css": "CSS", "nlp": "NLP",
    "c++": "C++", "c#": "C#", "c": "C", ".net": ".NET", "node.js": "Node.js",
    "next.js": "Next.js", "express.js": "Express.js", "ci/cd": "CI/CD",
    "rest api": "REST API", "mysql": "MySQL", "postgresql": "PostgreSQL",
    "mongodb": "MongoDB", "nosql": "NoSQL", "power bi": "Power BI", "pytorch": "PyTorch",
    "tensorflow": "TensorFlow", "javascript": "JavaScript", "typescript": "TypeScript",
    "github": "GitHub", "gitlab": "GitLab", "fastapi": "FastAPI", "graphql": "GraphQL",
    "opencv": "OpenCV", "ios": "iOS", "llm": "LLM", "etl": "ETL", "php": "PHP",
    "jquery": "jQuery", "xgboost": "XGBoost", "bigquery": "BigQuery",
    "scikit-learn": "scikit-learn", "golang": "Go", "ml": "ML", "k8s": "Kubernetes",
    "langchain": "LangChain", "hugging face": "Hugging Face", "generative ai": "Generative AI",
    "spring boot": "Spring Boot", "react native": "React Native", "jira": "Jira",
}


def canonical_skill(skill: str) -> str:
    key = skill.strip().lower()
    return ALIAS_TO_CANON.get(key, key)


def pretty_skill(skill: str) -> str:
    return PRETTY_SKILLS.get(skill, skill.title() if skill.islower() else skill)


@lru_cache(maxsize=4096)
def _skill_regex(skill: str) -> re.Pattern:
    names = {skill, *SKILL_ALIASES.get(skill, [])}
    alternatives = []
    for name in sorted(names, key=len, reverse=True):
        escaped = re.escape(name).replace(r"\ ", r"[\s\-]+").replace(" ", r"[\s\-]+")
        alternatives.append(escaped)
    core = "(?:" + "|".join(alternatives) + ")"
    guard = SKILL_GUARDS.get(skill, "")
    flags = re.IGNORECASE
    if skill == "c":
        # the language must be a capital "C" so ordinary text is not matched
        core, flags = "C", 0
    return re.compile(r"(?<![A-Za-z0-9+#])" + core + r"(?![A-Za-z0-9+#])" + guard, flags)


def find_skill(text: str, skill: str) -> re.Match | None:
    return _skill_regex(skill).search(text)


def extract_job_skills(job_description: str) -> list[str]:
    """Skills mentioned in the JD, ordered by first appearance."""
    found: list[tuple[int, str]] = []
    for skill in SKILL_ALIASES:
        match = find_skill(job_description, skill)
        if match:
            found.append((match.start(), skill))
    found.sort()
    names = [s for _, s in found]
    # "spring boot" should not also demand plain "spring" (only multi-word skills absorb others)
    kept = [
        s for s in names
        if not any(
            s != o and " " in o and re.search(rf"(?<![a-z0-9]){re.escape(s)}(?![a-z0-9])", o) for o in names
        )
    ]
    return kept


def match_skills(text: str, required: list[str]) -> tuple[list[str], list[str]]:
    matched = [s for s in required if find_skill(text, s)]
    missing = [s for s in required if s not in matched]
    return matched, missing


# ============================================================
# JOB DESCRIPTION PARSING
# ============================================================

_EXP_PATTERNS = [
    re.compile(r"(\d{1,2})\s*(?:-|–|to)\s*\d{1,2}\s*\+?\s*(?:years?|yrs?)", re.I),
    re.compile(r"(\d{1,2})\s*\+\s*(?:years?|yrs?)", re.I),
    re.compile(r"(?:minimum|min\.?|at least|atleast|over|more than)\s*(?:of\s*)?(\d{1,2})\s*\+?\s*(?:years?|yrs?)", re.I),
    re.compile(r"(\d{1,2})\s*(?:years?|yrs?)\s*(?:of\s+)?(?:[a-z\-]+\s+){0,3}?experience", re.I),
]


def extract_required_experience(job_description: str) -> int:
    best: tuple[int, int] | None = None
    for pattern in _EXP_PATTERNS:
        match = pattern.search(job_description)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), int(match.group(1)))
    return best[1] if best else 0


_EDU_PATTERNS = {
    "PhD": r"\bph\.?\s?d\b|\bdoctorate\b|\bdoctor of philosophy\b",
    "Masters": (
        r"\bm\.?\s?tech\b|\bm\.?\s?sc\b|\bm\.?\s?eng\b|\bmba\b|\bmca\b|\bpgdm\b|\bpost[- ]?graduate\b"
        r"|\bmaster'?s\b|\bmasters\b|\bmaster\s+(?:of|in|degree)\b|\bm\.\s?s\b"
    ),
    "Bachelors": (
        r"\bb\.?\s?tech\b|\bb\.\s?e\b|\bb\.?\s?sc\b|\bbca\b|\bbba\b|\bb\.?\s?com\b|\bb\.?\s?eng\b"
        r"|\bb\.\s?a\b|\bbachelor'?s?\b|\bundergraduate\b|\bgraduate degree\b|\bdegree in\b"
    ),
    "High School": r"\bhigh school\b|\bhsc\b|\b12th\b|\bsenior secondary\b|\bssc\b|\bged\b",
}
_EDU_REGEX = {k: re.compile(v, re.I) for k, v in _EDU_PATTERNS.items()}


def detect_education_levels(text: str) -> list[str]:
    return [level for level, rx in _EDU_REGEX.items() if rx.search(text)]


def extract_required_education(job_description: str) -> str:
    """Minimum level named in the JD ("B.Tech, B.E or MCA" -> Bachelors)."""
    levels = detect_education_levels(job_description)
    if not levels:
        return "Any"
    return min(levels, key=lambda lv: EDU_RANK[lv])


def highest_education(text: str) -> str:
    levels = detect_education_levels(text)
    if not levels:
        return "Unknown"
    return max(levels, key=lambda lv: EDU_RANK[lv])


# ============================================================
# RESUME TEXT EXTRACTION
# ============================================================

def _pdf_text(data: bytes) -> str:
    try:
        try:
            import pymupdf
        except ImportError:
            import fitz as pymupdf  # older PyMuPDF releases
    except ImportError:
        pymupdf = None

    if pymupdf is not None:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support needs `pip install pymupdf` (or pypdf).") from exc
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _docx_text(data: bytes) -> str:
    try:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:
        raise RuntimeError("DOCX support needs `pip install python-docx`.") from exc

    document = Document(io.BytesIO(data))
    lines: list[str] = []
    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            lines.append(Paragraph(child, document).text)
        elif child.tag.endswith("}tbl"):
            for row in Table(child, document).rows:
                seen, cells = [], []
                for cell in row.cells:           # merged cells repeat; skip the repeats
                    if any(cell._tc is s for s in seen):
                        continue
                    seen.append(cell._tc)
                    cells.append(cell.text.strip())
                lines.append(" | ".join(c for c in cells if c))
    return "\n".join(lines)


def extract_text(data: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        text = _pdf_text(data)
    elif ext == ".docx":
        text = _docx_text(data)
    elif ext in {".txt", ".md"}:
        text = data.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {ext or 'unknown'}")
    return text.replace("\u00a0", " ").replace("\x00", "")


# ============================================================
# RESUME FIELD EXTRACTION
# ============================================================

SECTION_HEADERS = {
    "experience": {
        "experience", "work experience", "professional experience", "employment history",
        "work history", "internships", "internship", "career history", "professional background",
        "employment", "internship experience",
    },
    "education": {
        "education", "academic background", "academics", "educational qualifications",
        "education qualification", "academic qualifications", "qualifications",
    },
    "projects": {
        "projects", "personal projects", "academic projects", "key projects", "project experience",
        "selected projects", "technical projects", "projects undertaken", "project work",
    },
    "skills": {
        "skills", "technical skills", "core competencies", "key skills", "skills summary",
        "tech stack", "technologies", "technical expertise",
    },
    "certifications": {"certifications", "certificates", "courses", "licenses", "training"},
    "achievements": {
        "achievements", "awards", "honors", "accomplishments", "publications",
        "extracurricular", "extracurriculars", "activities", "hobbies", "interests",
    },
    "summary": {"summary", "profile", "professional summary", "objective", "career objective", "about me", "about"},
}
IGNORED_FOR_EXPERIENCE = {"projects", "education", "certifications", "achievements", "skills"}


def _header_key(line: str) -> str | None:
    key = re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", line.lower())).strip()
    if not key or len(key) > 40:
        return None
    for section, names in SECTION_HEADERS.items():
        if key in names:
            return section
    return None


def iter_sections(text: str):
    """Yield (section, line) for every non-header line."""
    section = "header"
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        header = _header_key(line)
        if header:
            section = header
            continue
        yield section, line


_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_SEP = r"\s*(?:-|–|—|to|until|till)\s*"
_PRESENT = r"(present|current|currently|now|ongoing|till date|to date|date)"
_RANGE_MONTH_YEAR = re.compile(
    rf"\b{_MON}\.?,?\s+(\d{{4}}){_SEP}(?:{_MON}\.?,?\s+(\d{{4}})|{_PRESENT}\b)", re.I
)
_RANGE_NUMERIC = re.compile(
    rf"\b(\d{{1,2}})[/.](\d{{4}}){_SEP}(?:(\d{{1,2}})[/.](\d{{4}})|{_PRESENT}\b)", re.I
)
_RANGE_YEAR = re.compile(rf"\b((?:19|20)\d{{2}}){_SEP}(?:((?:19|20)\d{{2}})\b|{_PRESENT}\b)", re.I)
_EDU_CONTEXT = re.compile(
    r"school|college|university|institute|b\.?\s?tech|m\.?\s?tech|bachelor|master|degree|cgpa|gpa|percentage|%|"
    r"secondary|diploma|\bmba\b|\bmca\b|\bbca\b|ph\.?d",
    re.I,
)
_STATED_EXPERIENCE = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:[a-z\-]+\s+){0,2}?experience", re.I
)


def _month_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def extract_experience(text: str) -> tuple[float, str]:
    """Returns (years, source) where source is 'dates', 'stated' or 'none'."""
    now = datetime.now()
    now_idx = _month_index(now.year, now.month) + 1
    intervals: list[tuple[int, int]] = []

    def add(start: int, end: int) -> None:
        end = min(end, now_idx)
        if start < _month_index(1970, 1) or end <= start or start >= now_idx:
            return
        intervals.append((start, end))

    for section, line in iter_sections(text):
        if section in IGNORED_FOR_EXPERIENCE:
            continue
        low = line.lower()

        def blank(match: re.Match) -> str:
            return " " * len(match.group(0))

        for m in _RANGE_MONTH_YEAR.finditer(low):
            start = _month_index(int(m.group(2)), int(_month_number(m.group(1))))
            if m.group(3):
                end = _month_index(int(m.group(4)), _month_number(m.group(3))) + 1
            else:
                end = now_idx
            add(start, end)
        low = _RANGE_MONTH_YEAR.sub(blank, low)

        for m in _RANGE_NUMERIC.finditer(low):
            start_month, start_year = int(m.group(1)), int(m.group(2))
            if not 1 <= start_month <= 12:
                continue
            if m.group(3):
                end_month = int(m.group(3))
                if not 1 <= end_month <= 12:
                    continue
                end = _month_index(int(m.group(4)), end_month) + 1
            else:
                end = now_idx
            add(_month_index(start_year, start_month), end)
        low = _RANGE_NUMERIC.sub(blank, low)

        if not _EDU_CONTEXT.search(low):
            for m in _RANGE_YEAR.finditer(low):
                start = int(m.group(1)) * 12
                end = int(m.group(2)) * 12 if m.group(2) else now_idx
                add(start, end)

    if intervals:
        intervals.sort()
        merged = [list(intervals[0])]
        for start, end in intervals[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        months = sum(end - start for start, end in merged)
        return round(min(months / 12, FEATURE_LIMITS["years_experience"][1]), 2), "dates"

    stated = [float(m.group(1)) for m in _STATED_EXPERIENCE.finditer(text)]
    if stated:
        return round(min(max(stated), FEATURE_LIMITS["years_experience"][1]), 2), "stated"
    return 0.0, "none"


def _month_number(name: str) -> int:
    return ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"].index(name[:3].lower()) + 1


_BULLET = re.compile(r"^\s*(?:[\u2022\u25cf\u25aa\u25e6\u2023\u2043\u2219\u25ba\u27a2•▪◦●►➢*\-–—·»]|\d+[.)])\s*")
_PROJECT_NOISE = re.compile(r"^(tech(?:nologies| stack)?|tools?|stack|github|link|demo|url|description|role|built|developed|using)\b", re.I)
_REPO_LINK = re.compile(r"github\.com/[\w.-]+/[\w.-]+", re.I)


def count_projects(text: str) -> int:
    titles = bullets = 0
    for section, line in iter_sections(text):
        if section != "projects":
            continue
        if _BULLET.match(line):
            bullets += 1
            continue
        words = line.split()
        if len(words) <= 12 and not line.endswith(".") and not _PROJECT_NOISE.match(line) and line[0].isalnum():
            titles += 1
    if titles:
        count = titles
    elif bullets:
        count = math.ceil(bullets / 2)
    else:
        count = len({m.group(0).lower() for m in _REPO_LINK.finditer(text)})
    return int(min(count, FEATURE_LIMITS["project_count"][1]))


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<!\d)(\+?\d[\d\s().-]{8,16}\d)(?!\d)")
_LINKEDIN = re.compile(r"linkedin\.com/in/[\w%-]+", re.I)
_GITHUB_PROFILE = re.compile(r"github\.com/([\w-]+)", re.I)
_NOT_A_NAME = re.compile(r"resume|curriculum|vitae|\bcv\b|engineer|developer|analyst|manager|student|profile|summary|objective", re.I)


def extract_contact(text: str, filename: str) -> dict:
    email = _EMAIL.search(text)
    phone = next((m.group(1).strip() for m in _PHONE.finditer(text) if 10 <= len(re.sub(r"\D", "", m.group(1))) <= 13), "")
    linkedin = _LINKEDIN.search(text)
    github = _GITHUB_PROFILE.search(text)

    name = ""
    for line in [l.strip() for l in text.splitlines() if l.strip()][:6]:
        words = line.split()
        if 2 <= len(words) <= 4 and not re.search(r"[\d@:/|]", line) and not _NOT_A_NAME.search(line):
            if all(re.fullmatch(r"[A-Za-z.'\-]+", w) for w in words):
                name = line.title() if line.isupper() else line
                break
    if not name:
        name = re.sub(r"[_\-]+", " ", Path(filename).stem).strip().title() or filename

    return {
        "name": name,
        "email": email.group(0) if email else "",
        "phone": phone,
        "linkedin": linkedin.group(0) if linkedin else "",
        "github": f"github.com/{github.group(1)}" if github else "",
    }


@st.cache_data(show_spinner=False, max_entries=1024)
def parse_resume(data: bytes, filename: str) -> dict:
    """Everything that does not depend on the job description (cached per file)."""
    base = {"filename": filename, "error": "", "text": "", "text_hash": ""}
    try:
        text = extract_text(data, filename)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user in the UI
        return {**base, "error": str(exc), "name": Path(filename).stem}

    if len(text.strip()) < 80:
        return {
            **base,
            "error": "No readable text found (scanned image or empty file?). Try a text-based PDF/DOCX.",
            "name": Path(filename).stem,
        }

    experience, source = extract_experience(text)
    contact = extract_contact(text, filename)
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return {
        **base,
        **contact,
        "text": text,
        "text_hash": hashlib.md5(normalized.encode("utf-8")).hexdigest(),
        "words": len(re.findall(r"\w+", text)),
        "experience": experience,
        "experience_source": source,
        "education": highest_education(text),
        "projects": count_projects(text),
    }


# ============================================================
# MODEL  (load the pickle, otherwise retrain from the CSV)
# ============================================================

def _normalize_education_column(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.title()


def _build_pipeline():
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    preprocessor = ColumnTransformer(
        [
            ("education", OneHotEncoder(handle_unknown="ignore"), ["education_level"]),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", GradientBoostingClassifier(random_state=42))])


def _load_training_frame() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"{DATA_PATH.name} is missing columns: {sorted(missing)}")
    X = df[FEATURE_COLUMNS].copy()
    X["education_level"] = _normalize_education_column(X["education_level"])
    y = df[TARGET_COLUMN].astype(str).str.strip().str.lower().isin(["yes", "1", "true", "y"]).astype(int)
    return X, y


def _positive_index(model) -> int:
    classes = list(model.classes_)
    for candidate in (1, True, "Yes", "yes", "1"):
        if candidate in classes:
            return classes.index(candidate)
    return len(classes) - 1


def _education_map(model) -> dict[str, str]:
    """Lower-case label -> the exact category string the pipeline was fitted with."""
    try:
        categories = model.named_steps["preprocessor"].named_transformers_["education"].categories_[0]
        return {str(c).lower(): str(c) for c in categories}
    except Exception:  # noqa: BLE001
        return {"high school": "High School", "bachelors": "Bachelors", "masters": "Masters", "phd": "Phd"}


def _feature_importances(model) -> pd.Series | None:
    try:
        names = model.named_steps["preprocessor"].get_feature_names_out()
        values = model.named_steps["classifier"].feature_importances_
        series = pd.Series(values, index=[n.split("__", 1)[-1] for n in names])
        education = series[series.index.str.startswith("education_level")].sum()
        series = series[~series.index.str.startswith("education_level")]
        series["education_level"] = education
        return series.sort_values(ascending=False)
    except Exception:  # noqa: BLE001
        return None


@st.cache_resource(show_spinner="Preparing the screening model…")
def load_model_bundle() -> dict:
    bundle = {
        "model": None, "source": "", "notice": "", "metrics": None, "importances": None,
        "edu_map": {}, "pos_idx": 1, "data_summary": None,
    }
    probe = pd.DataFrame([{
        "years_experience": 3, "skills_match_score": 70.0, "education_level": "Bachelors",
        "project_count": 5, "resume_length": 500, "github_activity": 200,
    }])

    model = None
    load_error = ""
    if MODEL_PATH.exists():
        try:
            import joblib
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                candidate = joblib.load(MODEL_PATH)
                candidate.predict_proba(probe)           # fail early if it is not usable
            model = candidate
            bundle["source"] = f"Loaded {MODEL_PATH.name}"
        except Exception as exc:  # noqa: BLE001
            load_error = f"{type(exc).__name__}: {exc}"

    X = y = None
    if DATA_PATH.exists():
        try:
            X, y = _load_training_frame()
            bundle["data_summary"] = {
                "rows": int(len(X)),
                "shortlist_rate": float(y.mean()),
                "medians": X[NUMERIC_COLUMNS].median().to_dict(),
            }
        except Exception as exc:  # noqa: BLE001
            load_error = (load_error + " | " if load_error else "") + f"Training data: {exc}"

    if model is None and X is not None:
        try:
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            holdout = _build_pipeline().fit(X_train, y_train)
            predictions = holdout.predict(X_test)
            bundle["metrics"] = {
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1": f1_score(y_test, predictions, zero_division=0),
            }
            model = _build_pipeline().fit(X, y)
            bundle["source"] = "Retrained from ai_resume_screening.csv"
            bundle["notice"] = (
                "The saved model could not be used"
                + (f" ({load_error})" if load_error else "")
                + ", so it was retrained from the CSV with the installed scikit-learn."
            )
            try:
                import joblib
                joblib.dump(model, MODEL_PATH)
                bundle["notice"] += f" The refreshed model was saved to {MODEL_PATH.name}."
            except Exception:  # noqa: BLE001 - read-only folders are fine
                pass
        except Exception as exc:  # noqa: BLE001
            bundle["notice"] = f"Model unavailable ({type(exc).__name__}: {exc}). Scoring uses job-description fit only."
    elif model is None:
        bundle["notice"] = (
            "No usable model or training data found. Scoring uses job-description fit only."
            + (f" ({load_error})" if load_error else "")
        )

    if model is not None:
        bundle["model"] = model
        bundle["edu_map"] = _education_map(model)
        bundle["pos_idx"] = _positive_index(model)
        bundle["importances"] = _feature_importances(model)
    return bundle


def model_education(bundle: dict, label: str) -> str:
    """Translate an app label ('PhD', 'Unknown') into the pipeline's own category."""
    if label not in EDU_RANK or label == "Unknown":
        label = "Bachelors"   # modal class; flagged in the UI as an assumption
    return bundle["edu_map"].get(label.lower(), label)


def model_probabilities(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    X = frame[FEATURE_COLUMNS].copy()
    X["education_level"] = X["education_level"].map(lambda v: model_education(bundle, str(v)))
    for column, (low, high) in FEATURE_LIMITS.items():
        X[column] = pd.to_numeric(X[column], errors="coerce").fillna(low).clip(low, high)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bundle["model"].predict_proba(X)[:, bundle["pos_idx"]]


# ============================================================
# SCORING
# ============================================================

def clean_number(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else number


def score_candidates(parsed: list[dict], edits: pd.DataFrame, req: dict, cfg: dict, bundle: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for i, item in enumerate(parsed):
        edit = edits.iloc[i]
        matched, missing = match_skills(item["text"], req["skills"])
        skills_pct = 100.0 * len(matched) / len(req["skills"]) if req["skills"] else None
        experience = clean_number(edit["Experience (yrs)"], 0.0)
        education = edit["Education"] if edit["Education"] in EDU_RANK else "Unknown"
        projects = int(clean_number(edit["Projects"], 0))
        github = clean_number(edit["GitHub activity"], float(cfg["github_default"]))
        must_missing = [s for s in req["must"] if s in missing]

        exp_pct = 100.0 if req["exp"] <= 0 else min(experience / req["exp"], 1.0) * 100
        edu_ok = EDU_RANK[education] >= req["edu_rank"] and not (req["edu_rank"] > 0 and education == "Unknown")
        components = []
        if skills_pct is not None:
            components.append((cfg["w_skill"], skills_pct))
        if req["exp"] > 0:
            components.append((cfg["w_exp"], exp_pct))
        if req["edu_rank"] > 0:
            components.append((cfg["w_edu"], 100.0 if edu_ok else 0.0))
        total_weight = sum(w for w, _ in components)
        if components:
            rule_score = (
                sum(w * s for w, s in components) / total_weight
                if total_weight > 0 else sum(s for _, s in components) / len(components)
            )
        else:
            rule_score = None

        typed_name = edit["Candidate"]
        name = str(typed_name).strip() if pd.notna(typed_name) and str(typed_name).strip() else item["name"]
        rows.append({
            "idx": i,
            "candidate": name,
            "file": item["filename"],
            "email": item.get("email", ""),
            "phone": item.get("phone", ""),
            "linkedin": item.get("linkedin", ""),
            "github_profile": item.get("github", ""),
            "skills_pct": skills_pct,
            "matched": matched,
            "missing": missing,
            "must_missing": must_missing,
            "experience": experience,
            "experience_source": item.get("experience_source", "none"),
            "education": education,
            "edu_ok": edu_ok,
            "projects": projects,
            "github_activity": github,
            "words": item.get("words", 0),
            "rule_score": rule_score,
            "hr_decision": edit["HR decision"] if edit["HR decision"] in HR_DECISIONS else "Auto",
            "text_hash": item["text_hash"],
            "exp_pct": exp_pct,
        })

    df = pd.DataFrame(rows)
    df["rule_score"] = pd.to_numeric(df["rule_score"], errors="coerce")
    df["skills_pct"] = pd.to_numeric(df["skills_pct"], errors="coerce")

    if bundle["model"] is not None:
        features = pd.DataFrame({
            "years_experience": df["experience"],
            "skills_match_score": df["skills_pct"].fillna(NEUTRAL_SKILL_SCORE),
            "education_level": df["education"],
            "project_count": df["projects"],
            "resume_length": df["words"],
            "github_activity": df["github_activity"],
        })
        df["model_prob"] = model_probabilities(bundle, features)
    else:
        df["model_prob"] = np.nan

    alpha = cfg["ml_weight"] / 100 if bundle["model"] is not None else 0.0

    def blend(row) -> float:
        rule, prob = row["rule_score"], row["model_prob"]
        if pd.isna(rule):
            return 100 * prob if not pd.isna(prob) else 0.0
        if pd.isna(prob) or alpha == 0:
            return rule
        return (1 - alpha) * rule + alpha * 100 * prob

    df["final_score"] = df.apply(blend, axis=1).round(1)

    def decide(row) -> tuple[str, str]:
        score, must_ok = row["final_score"], not row["must_missing"]
        if score >= cfg["cutoff"]:
            auto = STATUS_SHORTLISTED if (must_ok or not cfg["strict_must"]) else STATUS_REVIEW
        elif score >= cfg["cutoff"] - cfg["band"]:
            auto = STATUS_REVIEW
        else:
            auto = STATUS_REJECTED
        if row["hr_decision"] == "Shortlist":
            return STATUS_SHORTLISTED, auto
        if row["hr_decision"] == "Reject":
            return STATUS_REJECTED, auto
        return auto, auto

    decisions = df.apply(decide, axis=1, result_type="expand")
    df["status"], df["auto_status"] = decisions[0], decisions[1]

    # flags
    first_seen: dict[str, str] = {}
    flags = []
    for _, row in df.iterrows():
        notes = []
        if row["text_hash"] in first_seen:
            notes.append(f"Duplicate of {first_seen[row['text_hash']]}")
        else:
            first_seen[row["text_hash"]] = row["candidate"]
        if row["experience_source"] == "none":
            notes.append("No experience dates found")
        elif row["experience_source"] == "stated":
            notes.append("Experience taken from a written claim, not dates")
        if row["education"] == "Unknown":
            notes.append("Education not detected")
        if row["hr_decision"] != "Auto":
            notes.append(f"HR override: {row['hr_decision']}")
        flags.append(notes)
    df["flags"] = flags

    df = df.sort_values(["final_score", "skills_pct"], ascending=[False, False], na_position="last").reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def explain(row: pd.Series, req: dict) -> tuple[list[str], list[str]]:
    good, gaps = [], []
    if req["skills"]:
        n, total = len(row["matched"]), len(req["skills"])
        (good if n / total >= 0.7 else gaps).append(f"Matches {n} of {total} required skills")
    if row["must_missing"]:
        gaps.append("Missing must-have: " + ", ".join(pretty_skill(s) for s in row["must_missing"]))
    if req["exp"] > 0:
        if row["experience"] >= req["exp"]:
            good.append(f"{row['experience']:.1f} years of experience meets the {req['exp']:g}+ requirement")
        else:
            gaps.append(f"{row['experience']:.1f} years of experience against {req['exp']:g} required")
    if req["edu_rank"] > 0:
        if row["edu_ok"]:
            good.append(f"{row['education']} meets the {REQ_EDU_OPTIONS[req['edu_rank']]} requirement")
        else:
            gaps.append(
                "Education not detected" if row["education"] == "Unknown"
                else f"{row['education']} is below the required {REQ_EDU_OPTIONS[req['edu_rank']]}"
            )
    prob = row["model_prob"]
    if pd.notna(prob):
        if prob >= 0.7:
            good.append(f"Profile resembles previously shortlisted candidates (model {prob:.0%})")
        elif prob < 0.4:
            gaps.append(f"Profile differs from previously shortlisted candidates (model {prob:.0%})")
    return good, gaps


# ============================================================
# UI HELPERS
# ============================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;500;600&display=swap');

:root {
  --ink: #1b2437; --muted: #5d6779; --canvas: #f3f5f8; --surface: #ffffff; --line: #dde2ea;
  --accent: #0e5a8a; --ok: #14713f; --ok-bg: #e3f3ea; --warn: #9a5b00; --warn-bg: #fcefd2;
  --bad: #b3261e; --bad-bg: #fbe5e3;
}
html, body, .stApp, [data-testid="stAppViewContainer"] { background: var(--canvas); color: var(--ink); font-family: 'Public Sans', system-ui, sans-serif; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
.block-container { padding-top: 2rem; max-width: 1180px; }

.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: 'Newsreader', Georgia, serif; color: var(--ink); letter-spacing: -0.01em; }
.stApp [data-testid="stMarkdownContainer"], .stApp label, .stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricLabel"], .stApp [data-testid="stCaptionContainer"] { color: var(--ink); }
.stApp [data-testid="stCaptionContainer"] { color: var(--muted); }
.stTextArea textarea, .stTextInput input, .stNumberInput input { background: #fff !important; color: var(--ink) !important; border-radius: 6px; }
div[data-baseweb="select"] > div { background: #fff !important; color: var(--ink) !important; }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); color: #fff; }
.stTabs [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding: 0.6rem 0.1rem; font-weight: 500; }

.rs-title { font-family: 'Newsreader', Georgia, serif; font-size: 2.3rem; font-weight: 600; line-height: 1.1; margin: 0; color: var(--ink); }
.rs-lede { color: var(--muted); margin: 0.35rem 0 1.4rem; max-width: 46rem; }

.rs-list { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.rs-row { display: grid; grid-template-columns: 2.4rem minmax(0, 1.5fr) minmax(0, 2fr) 8.5rem; gap: 1rem; align-items: center; padding: 0.85rem 1rem; border-bottom: 1px solid var(--line); }
.rs-row:last-child { border-bottom: 0; }
.rs-rank { font-family: 'Newsreader', Georgia, serif; font-size: 1.35rem; color: var(--muted); text-align: right; }
.rs-name { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rs-sub { color: var(--muted); font-size: 0.8rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rs-meter { min-width: 0; }
.rs-track { position: relative; height: 10px; background: #e8ecf2; border-radius: 5px; }
.rs-fill { position: absolute; inset: 0 auto 0 0; border-radius: 5px; }
.rs-fill.ok { background: var(--ok); } .rs-fill.warn { background: #d98e04; } .rs-fill.bad { background: #c9514a; }
.rs-cut { position: absolute; top: -4px; bottom: -4px; width: 2px; background: var(--ink); }
.rs-nums { display: flex; gap: 0.8rem; align-items: baseline; margin-top: 0.35rem; font-size: 0.8rem; color: var(--muted); flex-wrap: wrap; }
.rs-nums strong { font-size: 1.05rem; color: var(--ink); }
.rs-chip { justify-self: end; font-size: 0.8rem; font-weight: 600; padding: 0.25rem 0.65rem; border-radius: 999px; white-space: nowrap; }
.rs-chip.ok { background: var(--ok-bg); color: var(--ok); } .rs-chip.warn { background: var(--warn-bg); color: var(--warn); } .rs-chip.bad { background: var(--bad-bg); color: var(--bad); }
.rs-cutoff { display: flex; align-items: center; gap: 0.8rem; padding: 0.3rem 1rem; background: #eef1f6; color: var(--ink); font-size: 0.78rem; font-weight: 600; border-bottom: 1px solid var(--line); }
.rs-cutoff::before, .rs-cutoff::after { content: ""; flex: 1; border-top: 2px solid var(--ink); }

.rs-tag { display: inline-block; margin: 0 0.35rem 0.35rem 0; padding: 0.15rem 0.6rem; border-radius: 5px; font-size: 0.85rem; font-weight: 500; }
.rs-tag.ok { background: var(--ok-bg); color: var(--ok); } .rs-tag.bad { background: var(--bad-bg); color: var(--bad); } .rs-tag.neutral { background: #e8ecf2; color: var(--ink); }
.rs-panel { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 1rem 1.2rem; }
.rs-panel h5 { margin: 0 0 0.5rem; font-family: 'Public Sans', sans-serif; font-size: 0.9rem; color: var(--muted); font-weight: 600; }
.rs-panel ul { margin: 0; padding-left: 1.1rem; } .rs-panel li { margin-bottom: 0.3rem; }
.rs-contact { color: var(--muted); font-size: 0.9rem; margin: 0.2rem 0 0.8rem; }

@media (max-width: 760px) {
  .rs-row { grid-template-columns: 1.8rem minmax(0, 1fr); }
  .rs-meter, .rs-chip { grid-column: 2; justify-self: start; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
"""

_ST_VERSION = tuple(int(p) for p in re.findall(r"\d+", getattr(st, "__version__", "1.40"))[:2]) or (1, 40)


def stretch() -> dict:
    """Full-width kwarg that works on both old and new Streamlit releases."""
    return {"width": "stretch"} if _ST_VERSION >= (1, 50) else {"use_container_width": True}


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def tags_html(skills: list[str], kind: str) -> str:
    return "".join(f'<span class="rs-tag {kind}">{esc(pretty_skill(s))}</span>' for s in skills) or '<span class="rs-sub">None</span>'


def ranking_html(df: pd.DataFrame, cutoff: float) -> str:
    parts = ['<div class="rs-list">']
    cutoff_drawn = False
    cutoff_row = f'<div class="rs-cutoff">Cutoff {cutoff:g}</div>'
    for row in df.to_dict("records"):
        if not cutoff_drawn and row["final_score"] < cutoff:
            parts.append(cutoff_row)
            cutoff_drawn = True
        css = STATUS_CLASS[row["status"]]
        contact = row["email"] or row["phone"] or row["file"]
        detail = [f"JD fit {row['rule_score']:.0f}"] if pd.notna(row["rule_score"]) else []
        if pd.notna(row["model_prob"]):
            detail.append(f"Model {row['model_prob']:.0%}")
        if row["flags"]:
            detail.append(f"{len(row['flags'])} note{'s' if len(row['flags']) > 1 else ''}")
        width = max(0.0, min(row["final_score"], 100.0))
        parts.append(
            f'<div class="rs-row">'
            f'<div class="rs-rank">{row["rank"]}</div>'
            f'<div><div class="rs-name">{esc(row["candidate"])}</div><div class="rs-sub">{esc(contact)}</div></div>'
            f'<div class="rs-meter"><div class="rs-track"><div class="rs-fill {css}" style="width:{width:.1f}%"></div>'
            f'<div class="rs-cut" style="left:{cutoff:.1f}%"></div></div>'
            f'<div class="rs-nums"><strong>{row["final_score"]:.1f}</strong>' + "".join(f"<span>{esc(d)}</span>" for d in detail) + "</div></div>"
            f'<div class="rs-chip {css}">{esc(row["status"])}</div>'
            f"</div>"
        )
    if not cutoff_drawn:
        parts.append(cutoff_row)
    parts.append("</div>")
    return "".join(parts)


def short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def results_to_export(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "Rank": df["rank"],
        "Candidate": df["candidate"],
        "Status": df["status"],
        "Final score": df["final_score"],
        "JD fit": df["rule_score"].round(1),
        "Model probability": df["model_prob"].round(3),
        "Skill match %": df["skills_pct"].round(1),
        "Matched skills": df["matched"].map(lambda s: "; ".join(pretty_skill(x) for x in s)),
        "Missing skills": df["missing"].map(lambda s: "; ".join(pretty_skill(x) for x in s)),
        "Missing must-haves": df["must_missing"].map(lambda s: "; ".join(pretty_skill(x) for x in s)),
        "Experience (yrs)": df["experience"],
        "Education": df["education"],
        "Projects": df["projects"],
        "Email": df["email"],
        "Phone": df["phone"],
        "LinkedIn": df["linkedin"],
        "GitHub": df["github_profile"],
        "File": df["file"],
        "HR decision": df["hr_decision"],
        "Notes": df["flags"].map(lambda f: "; ".join(f)),
    })
    return out


# ============================================================
# UI SECTIONS
# ============================================================

def render_sidebar(bundle: dict) -> dict:
    model_ready = bundle["model"] is not None
    with st.sidebar:
        st.markdown("### Screening rules")
        cutoff = st.slider("Shortlist cutoff", 0, 100, 60, help="Candidates at or above this score are shortlisted.")
        band = st.slider("Review band", 0, 30, 10, help="Scores up to this many points below the cutoff are marked Review instead of rejected.")
        strict = st.checkbox("Missing a must-have caps the status at Review", value=True)

        st.markdown("#### Score weights")
        w_skill = st.slider("Skills", 0, 100, 50)
        w_exp = st.slider("Experience", 0, 100, 30)
        w_edu = st.slider("Education", 0, 100, 20)
        ml_weight = st.slider(
            "Model influence", 0, 100, 30 if model_ready else 0, disabled=not model_ready,
            help="Share of the final score that comes from the trained model. 0 means job-description fit only.",
        )
        with st.expander("Model assumptions"):
            github_default = st.number_input(
                "GitHub activity when unknown", 0, 842, DEFAULT_GITHUB_ACTIVITY,
                help="Resumes rarely state this number. The default is the training-data median so that "
                     "missing data neither helps nor hurts. Edit it per candidate in the review table.",
            )
        st.caption("Decision-support only. A person should make the final hiring decision.")
    return {
        "cutoff": cutoff, "band": band, "strict_must": strict, "w_skill": w_skill, "w_exp": w_exp,
        "w_edu": w_edu, "ml_weight": ml_weight, "github_default": github_default,
    }


def render_job_section() -> dict | None:
    st.markdown("#### 1. Job description")
    st.button("Use a sample job description", on_click=lambda: st.session_state.update(jd_text=SAMPLE_JD))
    jd = st.text_area(
        "Paste the job description", key="jd_text", height=220,
        placeholder="Include the skills, the years of experience and the education you require.",
    )
    if not jd.strip():
        return None

    jd_key = short_hash(jd)
    st.markdown("**Requirements detected.** Edit anything that looks wrong.")
    col_skills, col_exp, col_edu = st.columns([3, 1, 1.2])
    extracted_skills = extract_job_skills(jd)
    skills_text = col_skills.text_area(
        "Required skills (comma-separated)", value=", ".join(pretty_skill(s) for s in extracted_skills),
        key=f"skills_{jd_key}", height=110,
        help="Add or remove skills freely. Common abbreviations such as k8s or sklearn are understood.",
    )
    exp_req = col_exp.number_input(
        "Minimum experience (years)", 0.0, 30.0, float(extract_required_experience(jd)), 0.5, key=f"exp_{jd_key}"
    )
    detected_edu = extract_required_education(jd)
    edu_req = col_edu.selectbox(
        "Minimum education", REQ_EDU_OPTIONS, index=REQ_EDU_OPTIONS.index(detected_edu), key=f"edu_{jd_key}"
    )

    skills: list[str] = []
    for token in re.split(r"[,\n;]", skills_text):
        canon = canonical_skill(token)
        if canon and canon not in skills:
            skills.append(canon)

    must: list[str] = []
    if skills:
        must = st.multiselect(
            "Must-have skills (optional)", options=skills, format_func=pretty_skill,
            key=f"must_{short_hash(','.join(skills))}",
            help="A candidate missing any of these cannot be auto-shortlisted.",
        )
    else:
        st.warning("No skills detected. Add some above, otherwise the score relies on experience, education and the model.")
    return {"skills": skills, "must": must, "exp": float(exp_req), "edu_rank": REQ_EDU_OPTIONS.index(edu_req)}


def render_resume_section(cfg: dict) -> tuple[list[dict], pd.DataFrame | None]:
    st.markdown("#### 2. Resumes")
    files = st.file_uploader(
        "Upload resumes (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="resume_files"
    )
    if not files:
        return [], None

    payload = [(f.name, f.getvalue()) for f in files]
    signature = short_hash("|".join(f"{name}:{hashlib.md5(data).hexdigest()}" for name, data in payload))

    parsed: list[dict] = []
    progress = st.progress(0.0, text="Reading resumes…") if len(payload) > 3 else None
    for i, (name, data) in enumerate(payload, start=1):
        parsed.append(parse_resume(data, name))
        if progress:
            progress.progress(i / len(payload), text=f"Reading resumes… {i}/{len(payload)}")
    if progress:
        progress.empty()

    failed = [p for p in parsed if p["error"]]
    for item in failed:
        st.warning(f"{item['filename']}: {item['error']}")
    parsed = [p for p in parsed if not p["error"]]
    if not parsed:
        return [], None

    st.caption(f"{len(parsed)} resume{'s' if len(parsed) != 1 else ''} read.")
    base = pd.DataFrame({
        "Candidate": [p["name"] for p in parsed],
        "File": [p["filename"] for p in parsed],
        "Experience (yrs)": [float(p["experience"]) for p in parsed],
        "Education": [p["education"] for p in parsed],
        "Projects": [int(p["projects"]) for p in parsed],
        "GitHub activity": pd.Series([np.nan] * len(parsed), dtype="float64"),
        "HR decision": ["Auto"] * len(parsed),
    })
    with st.expander("Review what was extracted (editable)", expanded=len(parsed) <= 12):
        st.caption(
            "Extraction from free-form resumes is imperfect. Correct any value and the ranking updates. "
            "Leave GitHub activity empty to use the default. HR decision overrides the automatic status."
        )
        edited = st.data_editor(
            base, key=f"editor_{signature}", hide_index=True, num_rows="fixed", disabled=["File"],
            column_config={
                "Experience (yrs)": st.column_config.NumberColumn(min_value=0.0, max_value=40.0, step=0.5, format="%.1f"),
                "Education": st.column_config.SelectboxColumn(options=EDU_LEVELS, required=True),
                "Projects": st.column_config.NumberColumn(min_value=0, max_value=50, step=1),
                "GitHub activity": st.column_config.NumberColumn(
                    min_value=0, max_value=842, step=1, help="Empty = use the default from the sidebar"
                ),
                "HR decision": st.column_config.SelectboxColumn(options=HR_DECISIONS, required=True),
            },
            **stretch(),
        )
    return parsed, edited


def render_ranking_tab(results: pd.DataFrame | None, req: dict | None, cfg: dict) -> None:
    if req is None:
        st.info("Add a job description in the first tab.")
        return
    if results is None:
        st.info("Upload resumes in the first tab to see the shortlist.")
        return

    counts = results["status"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidates", len(results))
    c2.metric("Shortlisted", int(counts.get(STATUS_SHORTLISTED, 0)))
    c3.metric("Needs review", int(counts.get(STATUS_REVIEW, 0)))
    c4.metric("Median score", f"{results['final_score'].median():.1f}")

    view = st.radio("Show", ["All", STATUS_SHORTLISTED, STATUS_REVIEW, STATUS_REJECTED], horizontal=True, label_visibility="collapsed")
    shown = results if view == "All" else results[results["status"] == view]
    if shown.empty:
        st.info("No candidates in this group.")
    else:
        st.markdown(ranking_html(shown, cfg["cutoff"]), unsafe_allow_html=True)
        st.caption("The vertical mark on each bar is the shortlist cutoff.")

    with st.expander("Full table"):
        table = results_to_export(results)
        st.dataframe(
            table, hide_index=True,
            column_config={
                "Final score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
                "Skill match %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
                "Model probability": st.column_config.NumberColumn(format="%.2f"),
            },
            **stretch(),
        )

    export = results_to_export(results)
    left, right = st.columns(2)
    left.download_button(
        "Download full report (CSV)", export.to_csv(index=False).encode("utf-8"),
        file_name="resume_screening_report.csv", mime="text/csv", type="primary", **stretch(),
    )
    shortlist = export[export["Status"] == STATUS_SHORTLISTED]
    right.download_button(
        f"Download shortlist only ({len(shortlist)})", shortlist.to_csv(index=False).encode("utf-8"),
        file_name="shortlisted_candidates.csv", mime="text/csv", disabled=shortlist.empty, **stretch(),
    )


def render_report_tab(results: pd.DataFrame | None, req: dict | None, parsed: list[dict]) -> None:
    if req is None or results is None:
        st.info("Add a job description and upload resumes to see candidate reports.")
        return

    labels = {int(r["idx"]): f"#{r['rank']}  {r['candidate']}" for r in results.to_dict("records")}
    chosen = st.selectbox("Candidate", list(results["idx"].astype(int)), format_func=lambda i: labels[int(i)])
    row = results[results["idx"] == chosen].iloc[0]
    css = STATUS_CLASS[row["status"]]

    head, chip = st.columns([5, 1.4])
    head.markdown(f"### {esc(row['candidate'])}")
    chip.markdown(f'<div style="text-align:right;padding-top:1rem"><span class="rs-chip {css}">{esc(row["status"])}</span></div>', unsafe_allow_html=True)

    contact = [row["email"], row["phone"], row["linkedin"], row["github_profile"]]
    st.markdown(f'<div class="rs-contact">{esc("   ".join(c for c in contact if c) or "No contact details found")}</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final score", f"{row['final_score']:.1f}")
    m2.metric("Skills matched", "n/a" if pd.isna(row["skills_pct"]) else f"{len(row['matched'])}/{len(req['skills'])}")
    m3.metric("Experience", f"{row['experience']:.1f} yrs")
    m4.metric("Model", "n/a" if pd.isna(row["model_prob"]) else f"{row['model_prob']:.0%}")

    good, gaps = explain(row, req)
    col_good, col_gap = st.columns(2)
    col_good.markdown(
        '<div class="rs-panel"><h5>In favour</h5><ul>' + ("".join(f"<li>{esc(g)}</li>" for g in good) or "<li>Nothing stands out</li>") + "</ul></div>",
        unsafe_allow_html=True,
    )
    col_gap.markdown(
        '<div class="rs-panel"><h5>Gaps</h5><ul>' + ("".join(f"<li>{esc(g)}</li>" for g in gaps) or "<li>No gaps against the stated requirements</li>") + "</ul></div>",
        unsafe_allow_html=True,
    )

    st.markdown("")
    s1, s2 = st.columns(2)
    s1.markdown("**Skills found**")
    s1.markdown(tags_html(row["matched"], "ok"), unsafe_allow_html=True)
    s2.markdown("**Skills not found**")
    s2.markdown(tags_html(row["missing"], "bad"), unsafe_allow_html=True)

    if row["missing"]:
        st.markdown("**Questions to ask in the interview**")
        for skill in (row["must_missing"] + [s for s in row["missing"] if s not in row["must_missing"]])[:4]:
            st.markdown(f"- Ask about hands-on experience with {pretty_skill(skill)}. It was not found in the resume.")
    if row["flags"]:
        st.markdown("**Notes**")
        for note in row["flags"]:
            st.markdown(f"- {note}")

    with st.expander("Resume text as read by the system"):
        st.text_area("Extracted text", parsed[int(row["idx"])]["text"], height=320, disabled=True, label_visibility="collapsed")


def render_model_tab(bundle: dict, cfg: dict) -> None:
    st.markdown("#### The trained model")
    if bundle["model"] is None:
        st.warning(bundle["notice"] or "No model available.")
    else:
        st.write(f"**Source:** {bundle['source']}")
        if bundle["notice"]:
            st.info(bundle["notice"])
        if bundle["metrics"]:
            st.caption("Hold-out performance (20% of the CSV, not used for fitting):")
            cols = st.columns(len(bundle["metrics"]))
            for col, (name, value) in zip(cols, bundle["metrics"].items()):
                col.metric(name, f"{value:.1%}")
        summary = bundle["data_summary"]
        if summary:
            st.caption(f"Training data: {summary['rows']:,} candidates, {summary['shortlist_rate']:.0%} shortlisted.")
        if bundle["importances"] is not None:
            st.markdown("**What the model relies on**")
            st.bar_chart(bundle["importances"].rename("importance"))

        st.markdown("#### Playground")
        st.caption("Try a profile and see the model's shortlist probability.")
        medians = (summary or {}).get("medians", {})
        p1, p2, p3 = st.columns(3)
        years = p1.slider("Years of experience", 0, 15, int(medians.get("years_experience", 5)))
        skills_score = p2.slider("Skills match %", 0, 100, int(medians.get("skills_match_score", 74)))
        education = p3.selectbox("Education", EDU_LEVELS[1:], index=1)
        p4, p5, p6 = st.columns(3)
        projects = p4.slider("Projects", 0, 25, int(medians.get("project_count", 10)))
        words = p5.slider("Resume length (words)", 150, 900, int(medians.get("resume_length", 574)))
        github = p6.slider("GitHub activity", 0, 842, int(medians.get("github_activity", DEFAULT_GITHUB_ACTIVITY)))
        probe = pd.DataFrame([{
            "years_experience": years, "skills_match_score": skills_score, "education_level": education,
            "project_count": projects, "resume_length": words, "github_activity": github,
        }])
        st.metric("Shortlist probability", f"{model_probabilities(bundle, probe)[0]:.0%}")

        st.markdown("#### Score a structured CSV")
        st.caption("Upload a file with the columns: " + ", ".join(FEATURE_COLUMNS) + ".")
        batch = st.file_uploader("CSV file", type=["csv"], key="batch_csv", label_visibility="collapsed")
        if batch is not None:
            try:
                frame = pd.read_csv(batch)
                missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                else:
                    frame = frame.head(50000).copy()
                    frame["education_level"] = _normalize_education_column(frame["education_level"])
                    frame["education_level"] = frame["education_level"].replace({"Phd": "PhD"})
                    frame["shortlist_probability"] = model_probabilities(bundle, frame).round(4)
                    frame["prediction"] = np.where(frame["shortlist_probability"] >= 0.5, "Yes", "No")
                    st.dataframe(frame.head(200), hide_index=True, **stretch())
                    st.download_button("Download scored CSV", frame.to_csv(index=False).encode("utf-8"), "scored_candidates.csv", "text/csv")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not score that file: {exc}")

    with st.expander("How the score works"):
        st.markdown(
            """
**JD fit** is a weighted average of three parts: the share of required skills found in the resume,
how much of the required experience the candidate has (capped at 100%), and whether the education level
meets the minimum. Parts the job description does not specify are left out and the remaining weights are rescaled.

**Model score** is the probability from the trained pipeline, using six features: years of experience,
skills match, education, number of projects, resume length and GitHub activity.

**Final score** = (1 - model influence) × JD fit + model influence × model probability.

**Status** is Shortlisted at or above the cutoff, Review within the review band below it (or when a must-have skill is missing),
and Not shortlisted otherwise. An HR decision in the review table always wins.
"""
        )
    with st.expander("Limits worth knowing"):
        st.markdown(
            """
- The training data describes candidates with about 10 projects and 570-word resumes on average, which is more than many
  real resumes show. Probabilities can therefore look low for concise resumes. Lower the model influence if that happens.
- Resume length and GitHub activity are weak proxies for ability and can disadvantage some groups of candidates.
  The model sees them only as numeric inputs; no name, gender, age or photo is used.
- Skills are matched by keyword, so a skill described in other words will be missed. Check the Gaps list before rejecting anyone.
"""
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    st.set_page_config(page_title="Resume Screening", page_icon="📄", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("jd_text", "")

    bundle = load_model_bundle()
    cfg = render_sidebar(bundle)

    st.markdown('<p class="rs-title">Resume screening and shortlisting</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="rs-lede">Paste a job description, upload resumes, and get a ranked shortlist with the reasons behind each score.</p>',
        unsafe_allow_html=True,
    )

    tab_setup, tab_rank, tab_report, tab_model = st.tabs(["Job and resumes", "Shortlist", "Candidate report", "Model and scoring"])

    with tab_setup:
        req = render_job_section()
        st.markdown("")
        parsed, edits = render_resume_section(cfg)
        if bundle["model"] is None and bundle["notice"]:
            st.warning(bundle["notice"])

    results = None
    if req is not None and parsed and edits is not None:
        results = score_candidates(parsed, edits.reset_index(drop=True), req, cfg, bundle)

    with tab_rank:
        render_ranking_tab(results, req, cfg)
    with tab_report:
        render_report_tab(results, req, parsed)
    with tab_model:
        render_model_tab(bundle, cfg)


if __name__ == "__main__":
    main()