# app.py
print("🔥🔥🔥 RUNNING THIS APP.PY FILE 🔥🔥🔥")

import os
import json
import re
import base64
from datetime import datetime, timedelta, timezone
from io import BytesIO

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient
import bcrypt
import jwt
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import green, red, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from werkzeug.utils import secure_filename

# ===============================
# LOAD ENV
# ===============================
load_dotenv()

# ===============================
# TESSERACT CONFIG
# ===============================
import platform

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
else:
    # Linux/Render - tesseract is in PATH
    pytesseract.pytesseract.tesseract_cmd = "tesseract"
# ===============================
# FLASK APP
# ===============================
app = Flask(__name__)
CORS(app)

# ===============================
# UPLOAD CONFIG
# ===============================
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp"}

# ===============================
# MONGODB
# ===============================
try:
    mongo_client = MongoClient(os.getenv("MONGO_URI"))
    db = mongo_client.get_default_database()
    users_collection = db["users"]
    users_collection.create_index("email", unique=True)
    print("✅ MongoDB connected")
except Exception as e:
    print(f"❌ MongoDB connection error: {e}")

# ===============================
# OPENROUTER CLIENT
# ===============================
RENDER_URL = os.getenv(
    "RENDER_EXTERNAL_URL",
    "http://localhost:3000"
)

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": RENDER_URL,
        "X-Title": "AI Resume Analyzer",
    },
)

# ===============================
# HISTORY STORE (IN MEMORY)
# ===============================
history_store = []


# ===============================
# HELPER FUNCTIONS
# ===============================
def safe_parse(val):
    """Parse JSON string or return list."""
    if not val:
        return []
    if isinstance(val, list):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return []


def allowed_file(filename):
    """Check if file extension is allowed."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def get_file_extension(filename):
    """Get lowercase file extension."""
    return os.path.splitext(filename)[1].lower()


def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def extract_text_from_image(file_path):
    """Extract text from an image using OCR."""
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image, lang="eng")
    return text


def save_uploaded_file(file):
    """Save uploaded file and return path."""
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)
    return file_path


def cleanup_file(file_path):
    """Remove temporary uploaded file."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except OSError:
        pass


def extract_json_from_response(raw_text):
    """Extract JSON object from AI response text."""
    match = re.search(r'\{[\s\S]*\}', raw_text)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No JSON found in response")


def image_to_base64(file_path):
    """Convert image file to base64 string."""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_mime_type(filename):
    """Get MIME type for image."""
    ext = get_file_extension(filename)
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "image/jpeg")


# ===============================
# ROOT TEST
# ===============================
@app.route("/", methods=["GET"])
def root():
    return "Resume Analyzer Backend Running ✅"


# ===============================
# AUTH ROUTES
# ===============================
@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400

        existing = users_collection.find_one({"email": email})
        if existing:
            return jsonify({"error": "User already exists"}), 400

        hashed = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt()
        )

        users_collection.insert_one({
            "email": email,
            "password": hashed.decode("utf-8"),
        })

        return jsonify({"success": True})

    except Exception as e:
        print(f"Register error: {e}")
        return jsonify({"error": "Register failed"}), 500


@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"error": "Email and password required"}), 400

        user = users_collection.find_one({"email": email})
        if not user:
            return jsonify({"error": "Invalid credentials"}), 401

        match = bcrypt.checkpw(
            password.encode("utf-8"),
            user["password"].encode("utf-8")
        )

        if not match:
            return jsonify({"error": "Invalid credentials"}), 401

        token = jwt.encode(
            {
                "userId": str(user["_id"]),
                "exp": datetime.now(timezone.utc) + timedelta(days=7),
            },
            os.getenv("JWT_SECRET"),
            algorithm="HS256",
        )

        return jsonify({"success": True, "token": token})

    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({"error": "Login failed"}), 500


# ===============================
# ANALYZE
# ===============================
@app.route("/analyze", methods=["POST"])
def analyze():
    file_path = None
    try:
        print("📥 Analyze request received")

        job_description = request.form.get("jobDescription", "")

        if "resume" not in request.files:
            print("❌ No file in request")
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["resume"]
        print(f"📄 File received: {file.filename}")

        if not file.filename:
            return jsonify({"error": "No file uploaded"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Upload PDF or image"}), 400

        file_path = save_uploaded_file(file)
        print(f"💾 File saved to: {file_path}")

        ext = get_file_extension(file.filename)
        is_image = ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}

        if ext == ".pdf":
            text = extract_text_from_pdf(file_path)
            print(f"📝 Extracted text length: {len(text)}")

            if not text.strip():
                return jsonify({
                    "error": "Could not extract text from PDF"
                }), 400

            return analyze_with_text(text, job_description)

        elif is_image:
            print("🖼️ Image detected - using Vision AI")

            try:
                text = extract_text_from_image(file_path)
                print(f"📝 OCR extracted text length: {len(text)}")

                if text.strip() and len(text.strip()) > 100:
                    print("✅ Using OCR text for analysis")
                    return analyze_with_text(text, job_description)
            except Exception as ocr_error:
                print(f"⚠️ OCR failed: {ocr_error}")

            print("🔍 Using Vision AI to analyze image directly")
            return analyze_with_vision(file_path, file.filename, job_description)

        else:
            return jsonify({"error": "Upload PDF or image"}), 400

    except Exception as e:
        import traceback
        print(f"❌ Analyze error: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"Failed to analyze resume: {str(e)}"}), 500

    finally:
        if file_path:
            cleanup_file(file_path)


def analyze_with_text(text, job_description):
    """Analyze resume using extracted text."""
    print("🤖 Calling AI with text...")

    prompt = f"""
You are an advanced ATS resume analyzer and AI-detection system.

Resume Text:
{text[:4000]}

Job Description:
{job_description}

Analyze this resume and return STRICT JSON:

{{
  "atsScore": number (0-100),
  "fitScore": number (0-100),
  "skillStrength": {{"skill_name": number (1-5)}},
  "matchingSkills": ["skill1", "skill2"],
  "missingSkills": ["skill1", "skill2"],
  "aiDetection": {{
    "aiProbability": number (0-100),
    "riskLevel": "Low" | "Medium" | "High",
    "flaggedSections": [],
    "reasons": []
  }}
}}
"""

    completion = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are an ATS resume analyzer. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )

    raw = completion.choices[0].message.content
    print(f"🤖 AI Response: {raw[:500]}")

    parsed = extract_json_from_response(raw)
    print("✅ Parsed successfully")

    return jsonify({"success": True, **parsed})


def analyze_with_vision(file_path, filename, job_description):
    """Analyze resume image using Vision AI."""
    print("🔍 Converting image to base64...")

    base64_image = image_to_base64(file_path)
    mime_type = get_image_mime_type(filename)

    print(f"📊 Image base64 length: {len(base64_image)} chars")

    prompt = f"""
You are an advanced ATS resume analyzer with vision capabilities.

Look at this resume image and analyze it thoroughly.

Job Description:
{job_description if job_description else "General software developer position"}

Analyze the resume in the image and extract:
1. All skills mentioned
2. Work experience
3. Education
4. Projects
5. Overall quality

Then return STRICT JSON format:

{{
  "atsScore": number (0-100, based on resume quality, formatting, keywords),
  "fitScore": number (0-100, how well it matches the job description),
  "skillStrength": {{"skill_name": number (1-5)}},
  "matchingSkills": ["skill1", "skill2", "skill3"],
  "missingSkills": ["skill1", "skill2"],
  "aiDetection": {{
    "aiProbability": number (0-100),
    "riskLevel": "Low" | "Medium" | "High",
    "flaggedSections": [],
    "reasons": []
  }}
}}

Return ONLY the JSON, no other text.
"""

    print("🤖 Calling Vision AI...")

    completion = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        max_tokens=2000,
    )

    raw = completion.choices[0].message.content
    print(f"🤖 Vision AI Response: {raw[:500]}")

    parsed = extract_json_from_response(raw)
    print("✅ Vision analysis parsed successfully")

    return jsonify({"success": True, **parsed})


# ===============================
# RESUME OPTIMIZER
# ===============================
@app.route("/optimize-resume", methods=["POST"])
def optimize_resume():
    file_path = None
    try:
        job_description = request.form.get("jobDescription", "")

        if "resume" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["resume"]

        if not file.filename:
            return jsonify({"error": "No file uploaded"}), 400

        file_path = save_uploaded_file(file)
        ext = get_file_extension(file.filename)
        is_image = ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}

        if ext == ".pdf":
            text = extract_text_from_pdf(file_path)
            return optimize_with_text(text, job_description)
        elif is_image:
            try:
                text = extract_text_from_image(file_path)
                if text.strip() and len(text.strip()) > 100:
                    return optimize_with_text(text, job_description)
            except:
                pass
            return optimize_with_vision(file_path, file.filename, job_description)
        else:
            return jsonify({"error": "Upload PDF or image"}), 400

    except Exception as e:
        print(f"Optimize resume error: {e}")
        return jsonify({"error": "Failed to optimize resume"}), 500

    finally:
        if file_path:
            cleanup_file(file_path)


def optimize_with_text(text, job_description):
    """Optimize resume using text."""
    prompt = f"""
You are a professional resume writer.

Rewrite this resume to better match the job description.

Rules:
- Do NOT invent experience
- Improve wording
- Add missing keywords naturally
- ATS optimized
- Keep concise

Return STRICT JSON:

{{
  "optimizedSummary": "",
  "rewrittenExperience": [],
  "addedKeywords": []
}}

Resume:
{text[:3500]}

Job Description:
{job_description}
"""

    completion = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = completion.choices[0].message.content
    parsed = extract_json_from_response(raw)

    return jsonify({"success": True, **parsed})


def optimize_with_vision(file_path, filename, job_description):
    """Optimize resume using vision."""
    base64_image = image_to_base64(file_path)
    mime_type = get_image_mime_type(filename)

    prompt = f"""
You are a professional resume writer with vision capabilities.

Look at this resume image and rewrite it to better match the job description.

Job Description:
{job_description if job_description else "General software developer position"}

Rules:
- Do NOT invent experience
- Improve wording based on what you see
- Suggest missing keywords
- ATS optimized suggestions

Return STRICT JSON:

{{
  "optimizedSummary": "A better professional summary based on what you see",
  "rewrittenExperience": ["improved bullet point 1", "improved bullet point 2"],
  "addedKeywords": ["keyword1", "keyword2"]
}}

Return ONLY the JSON.
"""

    completion = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        temperature=0.3,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        max_tokens=2000,
    )

    raw = completion.choices[0].message.content
    parsed = extract_json_from_response(raw)

    return jsonify({"success": True, **parsed})


# ===============================
# INTERVIEW
# ===============================
@app.route("/interview", methods=["POST"])
def interview():
    file_path = None
    try:
        job_description = request.form.get("jobDescription", "")

        if "resume" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["resume"]

        if not file.filename:
            return jsonify({"error": "No file uploaded"}), 400

        file_path = save_uploaded_file(file)
        ext = get_file_extension(file.filename)
        is_image = ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}

        if ext == ".pdf":
            text = extract_text_from_pdf(file_path)
            return interview_with_text(text, job_description)
        elif is_image:
            try:
                text = extract_text_from_image(file_path)
                if text.strip() and len(text.strip()) > 100:
                    return interview_with_text(text, job_description)
            except:
                pass
            return interview_with_vision(file_path, file.filename, job_description)
        else:
            return jsonify({"error": "Upload PDF or image"}), 400

    except Exception as e:
        print(f"Interview error: {e}")
        return jsonify({"error": "Failed to generate interview prep"}), 500

    finally:
        if file_path:
            cleanup_file(file_path)


def interview_with_text(text, job_description):
    """Generate interview prep from text."""
    prompt = f"""
You are an interview coach.

Resume:
{text[:3000]}

Job Description:
{job_description}

Generate interview preparation in CLEAN TEXT FORMAT.

Include:

TECHNICAL QUESTIONS:
- 5 items

BEHAVIORAL QUESTIONS:
- 5 items

SYSTEM DESIGN PROMPTS:
- 3 items

CODING TOPICS:
- 5 items

Rules:
- NO JSON
- Use headings and bullet points only
- Human readable
"""

    completion = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "Interview coach"},
            {"role": "user", "content": prompt},
        ],
    )

    return jsonify({
        "success": True,
        "interviewPrep": completion.choices[0].message.content,
    })


def interview_with_vision(file_path, filename, job_description):
    """Generate interview prep from image."""
    base64_image = image_to_base64(file_path)
    mime_type = get_image_mime_type(filename)

    prompt = f"""
You are an interview coach with vision capabilities.

Look at this resume image and generate interview preparation.

Job Description:
{job_description if job_description else "General software developer position"}

Based on what you see in the resume, generate:

TECHNICAL QUESTIONS:
- 5 relevant questions based on their skills

BEHAVIORAL QUESTIONS:
- 5 questions based on their experience

SYSTEM DESIGN PROMPTS:
- 3 items relevant to their level

CODING TOPICS:
- 5 topics they should prepare

Rules:
- NO JSON
- Use headings and bullet points only
- Human readable format
"""

    completion = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        temperature=0.3,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        max_tokens=2000,
    )

    return jsonify({
        "success": True,
        "interviewPrep": completion.choices[0].message.content,
    })


# ===============================
# HISTORY
# ===============================
@app.route("/save-history", methods=["POST"])
def save_history():
    data = request.get_json()

    record = {
        "score": data.get("score"),
        "skills": data.get("skills", []),
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }

    history_store.insert(0, record)

    return jsonify({"success": True})


@app.route("/history", methods=["GET"])
def get_history():
    return jsonify(history_store)


# ===============================
# GENERATE PDF REPORT
# ===============================
@app.route("/generate-report", methods=["POST"])
def generate_report():
    try:
        score = request.form.get("score", "N/A")
        fit_score = request.form.get("fitScore", "N/A")
        skills = safe_parse(request.form.get("skills"))
        matching_skills = safe_parse(request.form.get("matchingSkills"))
        missing_skills = safe_parse(request.form.get("missingSkills"))
        interview_prep = request.form.get("interviewPrep", "")

        buffer = BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=40,
            rightMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Title"],
            fontSize=24,
            alignment=1,
        )

        heading_style = ParagraphStyle(
            "CustomHeading",
            parent=styles["Heading2"],
            fontSize=18,
        )

        normal_style = styles["Normal"]

        green_style = ParagraphStyle(
            "GreenText",
            parent=styles["Normal"],
            textColor=green,
        )

        red_style = ParagraphStyle(
            "RedText",
            parent=styles["Normal"],
            textColor=red,
        )

        elements = []

        elements.append(Paragraph("AI Resume Analysis Report", title_style))
        elements.append(Spacer(1, 12))

        elements.append(
            Paragraph(
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                normal_style,
            )
        )
        elements.append(Spacer(1, 24))

        elements.append(Paragraph("Summary Scores", heading_style))
        elements.append(Spacer(1, 8))
        elements.append(Paragraph(f"ATS Score: {score}%", normal_style))
        elements.append(Paragraph(f"Fit Score: {fit_score}%", normal_style))
        elements.append(Spacer(1, 24))

        elements.append(Paragraph("Skills Found", heading_style))
        elements.append(Spacer(1, 8))
        for s in skills:
            elements.append(Paragraph(f"• {s}", normal_style))
        elements.append(Spacer(1, 16))

        elements.append(Paragraph("Matching Skills", heading_style))
        elements.append(Spacer(1, 8))
        for s in matching_skills:
            elements.append(Paragraph(f"✔ {s}", green_style))
        elements.append(Spacer(1, 16))

        elements.append(Paragraph("Missing Skills", heading_style))
        elements.append(Spacer(1, 8))
        for s in missing_skills:
            elements.append(Paragraph(f"✖ {s}", red_style))
        elements.append(Spacer(1, 16))

        if interview_prep:
            elements.append(Paragraph("Interview Preparation", heading_style))
            elements.append(Spacer(1, 8))
            for line in interview_prep.split("\n"):
                if line.strip():
                    elements.append(Paragraph(line, normal_style))
                    elements.append(Spacer(1, 4))

        doc.build(elements)
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="resume_report.pdf",
        )

    except Exception as e:
        print(f"PDF ERROR: {e}")
        return jsonify({"error": "Failed to generate PDF"}), 500

# ===============================
# COVER LETTER GENERATOR
# ===============================
@app.route("/generate-cover-letter", methods=["POST"])
def generate_cover_letter():
    file_path = None
    try:
        print("📧 Cover letter request received")

        job_description = request.form.get("jobDescription", "")
        company_name = request.form.get("companyName", "the company")
        hiring_manager = request.form.get("hiringManager", "Hiring Manager")

        if "resume" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["resume"]
        file_path = save_uploaded_file(file)

        ext = get_file_extension(file.filename)
        is_image = ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}

        # Get resume content
        if ext == ".pdf":
            text = extract_text_from_pdf(file_path)
            if text.strip():
                return generate_cover_letter_text(
                    text, job_description, company_name, hiring_manager
                )

        if is_image or not text.strip():
            # Try OCR first
            try:
                text = extract_text_from_image(file_path)
                if text.strip() and len(text.strip()) > 100:
                    return generate_cover_letter_text(
                        text, job_description, company_name, hiring_manager
                    )
            except:
                pass

            # Use vision
            return generate_cover_letter_vision(
                file_path, file.filename, job_description, company_name, hiring_manager
            )

        return generate_cover_letter_text(
            text, job_description, company_name, hiring_manager
        )

    except Exception as e:
        import traceback
        print(f"❌ Cover letter error: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "Failed to generate cover letter"}), 500

    finally:
        if file_path:
            cleanup_file(file_path)


def generate_cover_letter_text(text, job_description, company_name, hiring_manager):
    """Generate cover letter from resume text."""
    print("📝 Generating cover letter from text...")

    prompt = f"""
You are an expert cover letter writer with years of experience helping candidates land jobs.

Based on this resume and job description, write a compelling, professional cover letter.

RESUME:
{text[:3500]}

JOB DESCRIPTION:
{job_description if job_description else "General professional position"}

COMPANY NAME: {company_name}
HIRING MANAGER: {hiring_manager}

RULES:
1. Professional but warm and personable tone
2. Strong opening that grabs attention
3. Highlight 2-3 most relevant experiences/achievements from resume
4. Show genuine enthusiasm for the role and company
5. Include specific examples with metrics if available
6. Strong closing with call to action
7. Keep it concise - 3-4 paragraphs maximum
8. Do NOT copy resume bullet points directly - reframe them
9. Make it feel authentic, not generic

FORMAT:
- Start with "Dear {hiring_manager},"
- End with "Sincerely," and leave space for name

Return ONLY the cover letter text, no additional commentary.
"""

    completion = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": "You are an expert cover letter writer. Write compelling, personalized cover letters."
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
    )

    cover_letter = completion.choices[0].message.content
    print("✅ Cover letter generated")

    return jsonify({
        "success": True,
        "coverLetter": cover_letter,
    })


def generate_cover_letter_vision(file_path, filename, job_description, company_name, hiring_manager):
    """Generate cover letter from resume image."""
    print("🔍 Generating cover letter from image...")

    base64_image = image_to_base64(file_path)
    mime_type = get_image_mime_type(filename)

    prompt = f"""
You are an expert cover letter writer.

Look at this resume image and write a compelling, professional cover letter.

JOB DESCRIPTION:
{job_description if job_description else "General professional position"}

COMPANY NAME: {company_name}
HIRING MANAGER: {hiring_manager}

RULES:
1. Professional but warm tone
2. Strong attention-grabbing opening
3. Highlight 2-3 most relevant experiences from what you see
4. Show enthusiasm for the role
5. Include specific examples if visible
6. Strong closing with call to action
7. 3-4 paragraphs maximum
8. Make it authentic, not generic

FORMAT:
- Start with "Dear {hiring_manager},"
- End with "Sincerely," and leave space for name

Return ONLY the cover letter text.
"""

    completion = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        temperature=0.5,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        max_tokens=1500,
    )

    cover_letter = completion.choices[0].message.content
    print("✅ Cover letter generated from image")

    return jsonify({
        "success": True,
        "coverLetter": cover_letter,
    })


# ===============================
# LINKEDIN OPTIMIZER
# ===============================
@app.route("/optimize-linkedin", methods=["POST"])
def optimize_linkedin():
    file_path = None
    try:
        print("💼 LinkedIn optimization request received")

        job_description = request.form.get("jobDescription", "")
        target_role = request.form.get("targetRole", "")

        if "resume" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["resume"]
        file_path = save_uploaded_file(file)

        ext = get_file_extension(file.filename)
        is_image = ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}

        # Get resume content
        if ext == ".pdf":
            text = extract_text_from_pdf(file_path)
            if text.strip():
                return optimize_linkedin_text(text, job_description, target_role)

        if is_image or not text.strip():
            try:
                text = extract_text_from_image(file_path)
                if text.strip() and len(text.strip()) > 100:
                    return optimize_linkedin_text(text, job_description, target_role)
            except:
                pass

            return optimize_linkedin_vision(
                file_path, file.filename, job_description, target_role
            )

        return optimize_linkedin_text(text, job_description, target_role)

    except Exception as e:
        import traceback
        print(f"❌ LinkedIn optimization error: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "Failed to optimize LinkedIn"}), 500

    finally:
        if file_path:
            cleanup_file(file_path)


def optimize_linkedin_text(text, job_description, target_role):
    """Optimize LinkedIn from resume text."""
    print("📝 Optimizing LinkedIn from text...")

    prompt = f"""
You are a LinkedIn optimization expert who has helped thousands of professionals improve their profiles.

Based on this resume, generate optimized LinkedIn profile content.

RESUME:
{text[:3500]}

TARGET ROLE/INDUSTRY:
{target_role if target_role else job_description if job_description else "General professional"}

Generate the following LinkedIn content:

1. HEADLINE (max 220 characters)
   - Keyword-rich but readable
   - Include value proposition
   - Example format: "Role | Specialty | Value Prop" or "Helping X do Y through Z"

2. ABOUT SECTION (max 2600 characters)
   - Hook in first 2 lines (visible before "see more")
   - Tell your professional story
   - Highlight key achievements with metrics
   - Include relevant keywords naturally
   - End with call to action
   - Use short paragraphs and line breaks

3. EXPERIENCE BULLETS (for each role)
   - Start with action verbs
   - Include metrics and results
   - Show impact, not just duties
   - Keyword optimized

4. FEATURED SKILLS (top 10)
   - Most relevant to target role
   - Mix of hard and soft skills
   - Endorsable skills

5. KEYWORDS TO ADD
   - Industry-specific terms
   - Tools and technologies
   - Certifications if any

6. PROFILE TIPS
   - Specific suggestions to improve their profile

Return STRICT JSON:

{{
  "headline": "Your optimized headline here",
  "about": "Your optimized about section here...",
  "experienceBullets": [
    {{
      "company": "Company Name",
      "role": "Job Title", 
      "bullets": ["Achievement 1", "Achievement 2", "Achievement 3"]
    }}
  ],
  "featuredSkills": ["Skill 1", "Skill 2", "Skill 3", "Skill 4", "Skill 5"],
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "profileTips": ["Tip 1", "Tip 2", "Tip 3"]
}}
"""

    completion = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": "You are a LinkedIn optimization expert. Return only valid JSON."
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=2500,
    )

    raw = completion.choices[0].message.content
    print(f"🤖 LinkedIn Response: {raw[:500]}")

    parsed = extract_json_from_response(raw)
    print("✅ LinkedIn optimization complete")

    return jsonify({"success": True, **parsed})


def optimize_linkedin_vision(file_path, filename, job_description, target_role):
    """Optimize LinkedIn from resume image."""
    print("🔍 Optimizing LinkedIn from image...")

    base64_image = image_to_base64(file_path)
    mime_type = get_image_mime_type(filename)

    prompt = f"""
You are a LinkedIn optimization expert.

Look at this resume image and generate optimized LinkedIn profile content.

TARGET ROLE/INDUSTRY:
{target_role if target_role else job_description if job_description else "General professional"}

Generate:

1. HEADLINE (max 220 chars) - Keyword-rich, includes value proposition
2. ABOUT SECTION - Hook, story, achievements, keywords, CTA
3. EXPERIENCE BULLETS - Action verbs, metrics, impact
4. FEATURED SKILLS - Top 10 relevant skills
5. KEYWORDS - Industry-specific terms
6. PROFILE TIPS - Specific improvement suggestions

Return STRICT JSON:

{{
  "headline": "string",
  "about": "string",
  "experienceBullets": [
    {{
      "company": "Company Name",
      "role": "Job Title",
      "bullets": ["bullet1", "bullet2"]
    }}
  ],
  "featuredSkills": ["skill1", "skill2"],
  "keywords": ["keyword1", "keyword2"],
  "profileTips": ["tip1", "tip2"]
}}

Return ONLY the JSON.
"""

    completion = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        temperature=0.3,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        max_tokens=2500,
    )

    raw = completion.choices[0].message.content
    parsed = extract_json_from_response(raw)
    print("✅ LinkedIn optimization from image complete")

    return jsonify({"success": True, **parsed})

# ===============================
# START SERVER
# ===============================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    is_production = os.getenv("RENDER") is not None

    if is_production:
        print(f"🚀 Server running in PRODUCTION on port {port}")
    else:
        print(f"🚀 Server running on http://localhost:{port}")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=not is_production,
    )