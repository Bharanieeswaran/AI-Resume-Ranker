# app.py (232 lines with updated compare flow using session)
import os
import fitz  # PyMuPDF
import docx2txt
import spacy
import pandas as pd
import datetime
import re
from flask import Flask, request, render_template, send_file, redirect, url_for, session
from sklearn.feature_extraction.text import TfidfVectorizer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)
app.secret_key = 'super_secret'

UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

nlp = spacy.load("en_core_web_sm")
users = {"admin": "admin123"}  # Demo user database

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(docx_path):
    return docx2txt.process(docx_path)

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

def extract_sections(text):
    sections = {
        'Name': '', 'Email': '', 'Phone': '',
        'Skills': '', 'Education': '', 'Experience': ''
    }
    email_match = re.search(r"[\w.-]+@[\w.-]+", text)
    phone_match = re.search(r"\b\d{10}\b", text)
    name_match = text.strip().split("\n")[0]
    sections['Email'] = email_match.group(0) if email_match else "Not Found"
    sections['Phone'] = phone_match.group(0) if phone_match else "Not Found"
    sections['Name'] = name_match.strip()
    lower_text = text.lower()
    if 'skills' in lower_text:
        sections['Skills'] = 'Found'
    if 'education' in lower_text:
        sections['Education'] = 'Found'
    if 'experience' in lower_text:
        sections['Experience'] = 'Found'
    return sections

def get_skill_gap(job_desc, resume_text):
    jd_tokens = set(preprocess(job_desc).split())
    resume_tokens = set(preprocess(resume_text).split())
    return list(jd_tokens - resume_tokens)

def generate_feedback(missing_skills):
    if not missing_skills:
        return "Great match! Resume covers all required areas."
    if len(missing_skills) < 5:
        return f"Minor gaps found. Try including: {', '.join(missing_skills[:5])}"
    return f"Several key skills missing. Consider adding: {', '.join(missing_skills[:7])}"

def rank_resumes(job_description, resume_texts, resume_names):
    job_keywords = set(preprocess(job_description).split())
    matched_keywords_list = []

    processed_texts = [preprocess(job_description)] + [preprocess(text) for text in resume_texts]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    scores = (tfidf_matrix[1:] @ tfidf_matrix[0].T).toarray().flatten()

    for text in processed_texts[1:]:
        resume_words = set(text.split())
        matched = job_keywords.intersection(resume_words)
        matched_keywords_list.append(", ".join(sorted(matched)))

    ranked = sorted(zip(resume_names, scores, matched_keywords_list), key=lambda x: x[1], reverse=True)
    return ranked

def create_pdf_report(data, filepath):
    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30, height - 40, "AI Resume Ranking Report")
    c.setFont("Helvetica", 11)
    y = height - 80
    for i, (name, score, keywords) in enumerate(data, 1):
        if y < 80:
            c.showPage()
            y = height - 50
        c.drawString(30, y, f"{i}. {name} - Score: {round(score, 4)}")
        c.drawString(50, y - 15, f"Matched Keywords: {keywords}")
        y -= 40
    c.save()

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        users[request.form['username']] = request.form['password']
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/forgot', methods=['GET'])
def forgot():
    return render_template('forgot.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']
        if u in users and users[u] == p:
            session['logged_in'] = True
            return redirect(url_for('index'))
        return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        job_desc = request.form['job_desc']
        resumes = request.files.getlist('resumes')

        resume_texts, resume_names = [], []
        for resume in resumes:
            filename = resume.filename
            path = os.path.join(UPLOAD_FOLDER, filename)
            resume.save(path)
            if filename.endswith(".pdf"):
                text = extract_text_from_pdf(path)
            elif filename.endswith(".docx"):
                text = extract_text_from_docx(path)
            else:
                text = ""
            resume_texts.append(text)
            resume_names.append(filename)

        ranked = rank_resumes(job_desc, resume_texts, resume_names)

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = f"ranked_{timestamp}.csv"
        pdf_file = f"ranked_{timestamp}.pdf"
        csv_path = os.path.join(REPORT_FOLDER, csv_file)
        pdf_path = os.path.join(REPORT_FOLDER, pdf_file)

        pd.DataFrame(ranked, columns=["Resume", "Score", "Matched Keywords"]).to_csv(csv_path, index=False)
        create_pdf_report(ranked, pdf_path)

        return render_template('index.html', ranked=ranked, report_link=csv_file, pdf_link=pdf_file)

    return render_template('index.html')

@app.route('/reports/<filename>')
def download_file(filename):
    return send_file(os.path.join(REPORT_FOLDER, filename), as_attachment=True)

@app.route('/check_rank', methods=['GET', 'POST'])
def check_rank():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    if request.method == 'POST':
        job_desc = request.form['job_desc']
        resume = request.files['resume']
        filename = resume.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        resume.save(path)
        text = extract_text_from_pdf(path) if filename.endswith(".pdf") else extract_text_from_docx(path)
        jd = preprocess(job_desc)
        rt = preprocess(text)
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([jd, rt])
        score = (tfidf[1] @ tfidf[0].T).toarray()[0][0]
        matched = ", ".join(sorted(set(jd.split()) & set(rt.split())))
        gap = get_skill_gap(job_desc, text)
        fb = generate_feedback(gap)
        section = extract_sections(text)
        return render_template('check_rank.html', score=score, matched=matched, gap=gap, feedback=fb, sections=section)
    return render_template('check_rank.html')

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        job_desc = request.form['job_desc']
        resume = request.files['resume']
        filename = resume.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        resume.save(path)
        text = extract_text_from_pdf(path) if filename.endswith(".pdf") else extract_text_from_docx(path)

        jd = preprocess(job_desc)
        rt = preprocess(text)
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([jd, rt])
        score = (tfidf[1] @ tfidf[0].T).toarray()[0][0]
        matched = ", ".join(sorted(set(jd.split()) & set(rt.split())))
        gap = get_skill_gap(job_desc, text)
        fb = generate_feedback(gap)
        section = extract_sections(text)

        # Store in session
        if 'resume1' not in session:
            session['resume1'] = {
                'score': score, 'matched': matched, 'gap': gap,
                'feedback': fb, 'section': section, 'job_desc': job_desc
            }
            return render_template('compare.html', res1=session['resume1'])
        else:
            res1 = session.pop('resume1')
            res2 = {
                'score': score, 'matched': matched, 'gap': gap,
                'feedback': fb, 'section': section
            }
            return render_template('compare.html', res1=res1, res2=res2)

    return render_template('compare.html')

if __name__ == '__main__':
    app.run(debug=True)
