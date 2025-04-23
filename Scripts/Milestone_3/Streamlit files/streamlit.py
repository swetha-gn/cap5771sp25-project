# Gemini Career Assistant (Streamlit Version with Profile Richness)

import streamlit as st
import pandas as pd
import joblib
import re
import pdfplumber
import google.generativeai as genai

#cc
import joblib
cc_model = joblib.load("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Coursera/best_course_popularity_model.pkl")

import pandas as pd
df_cc = pd.read_csv("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Coursera/cleaned_coursera_data.csv")

#jd
jdc_model = joblib.load("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Job_descriptions/random_forest_classifier.pkl")
jdr_model = joblib.load("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Job_descriptions/random_forest_regressor.pkl")
df_jd= pd.read_csv("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Job_descriptions/df_chatbot.csv")

#ll

ll_model = joblib.load("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Linkedin/best_profile_tier_model.pkl")
df_ll = pd.read_csv("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Linkedin/preprocessed_linkedin_data.csv")

#rs
rs_model = joblib.load("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Roles_based_skills/best_model.joblib")
df_rs= pd.read_csv("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Roles_based_skills/final_role_predictions.csv")

#  Gemini Configuration 
genai.configure(api_key="AIzaSyDTo5J7P130C5cUbqwcUADHtb6MXj1_2ms")
model = genai.GenerativeModel("models/gemini-1.5-flash")



#  Resume Parsing 
SECTION_HEADERS = {'skills': ['skills', 'technical skills', 'key skills', 'technical summary', 'highlights']}

def extract_text_from_pdf_bytes(file_obj):
    with pdfplumber.open(file_obj) as pdf:
        return ' '.join([page.extract_text() or "" for page in pdf.pages])

def extract_sections(text):
    lines = text.split('\n')
    sections = {}
    current_section = None
    buffer = []
    for line in lines:
        for key, variants in SECTION_HEADERS.items():
            if any(re.match(rf"^\s*{v}\s*$", line.strip(), re.IGNORECASE) for v in variants):
                if current_section and buffer:
                    sections[current_section] = '\n'.join(buffer).strip()
                current_section = key
                buffer = []
                break
        else:
            if current_section:
                buffer.append(line)
    if current_section and buffer:
        sections[current_section] = '\n'.join(buffer).strip()
    return sections

def extract_skills(text):
    skills = set()
    for line in text.splitlines():
        line = re.sub(r"[^\w\s,()/+.-]", "", line)
        for token in re.split(r",| and | or | with ", line):
            token = token.strip().lower()
            if len(token) > 1 and not token.isdigit():
                skills.add(token)
    return skills

def compute_profile_features(parsed, matched_skills):
    exp_text = parsed.get("work_experience", "")
    cert_text = parsed.get("certifications", "")
    seniority_text = parsed.get("summary", "") + parsed.get("profile", "")

    experience_years = len(re.findall(r"\d+\+?\s+years?", exp_text.lower())) or 1
    skill_count = len(matched_skills)
    cert_count = len(re.findall(r"\b(certification|certificate|course)\b", cert_text.lower()))
    seniority_score = len(re.findall(r"senior|lead|manager|head", seniority_text.lower()))
    exp_skill_ratio = round(experience_years / skill_count, 2) if skill_count else 0.0

    df = pd.DataFrame.from_records([{
        "Experience_Years": experience_years,
        "Skill_Count": skill_count,
        "Cert_Count": cert_count,
        "Seniority_Score": seniority_score,
        "Exp_Skill_Ratio": exp_skill_ratio
    }])
    return df[["Experience_Years", "Skill_Count", "Cert_Count", "Seniority_Score", "Exp_Skill_Ratio"]]

#  Streamlit App 
st.set_page_config(page_title="Gemini Career Assistant", layout="centered")
st.title("💼 Gemini Career Assistant")

name = st.text_input("What's your name?")
if name:
    current_role = st.selectbox("💼 What's your current role?", options=["Student", "Intern", "Web Developer", "Data Scientist", "Other"])
    desired_role = st.text_input("What role are you aiming for?")

    if desired_role:
        if st.toggle("📍 Show top hiring locations for this role"):
            st.subheader("🌍 Top Hiring Locations")
            top_locs = df_jd[df_jd['Role'].str.contains(desired_role, case=False, na=False)]['Country'].value_counts().head(3)
            st.dataframe(top_locs)

        st.subheader(" Common Skills for This Role")
        skill_prompt = f"List 10 key technical and soft skills needed for the role: {desired_role}. Pretend it was extracted from a real job dataset."
        st.write(model.generate_content(skill_prompt).text)

        if current_role.lower() in ["student", "intern"] or current_role.lower() != desired_role.lower():
            st.subheader("🎓 Career Switch Guidance")
            df_cc['combined'] = df_cc['title'] + ' ' + df_cc['Skills']
            courses = df_cc[df_cc['combined'].str.contains(desired_role.split()[0], case=False, na=False)][['title', 'URL']].head(3)
            st.write("Recommended Coursera Courses:")
            st.dataframe(courses)

            pros = df_ll[df_ll['Current_Role'].str.contains(desired_role.split()[0], case=False, na=False)][['Full_Name', 'Contact_mail']].dropna().head(3)
            st.write("👥 Professionals You Can Reach Out To:")
            st.dataframe(pros)

            msg = model.generate_content(f"{name} is new to the {desired_role} field. Suggest a short message asking professionals for guidance and learning resources.").text
            st.write("📨 Suggested Message:")
            st.markdown(msg)

        else:
            st.subheader("📄 Upload Resume")
            uploaded_file = st.file_uploader("Upload your resume as PDF", type="pdf")

            if uploaded_file:
                text = extract_text_from_pdf_bytes(uploaded_file)
                parsed = extract_sections(text)
                skill_text = parsed.get("skills", "") or text
                extracted = extract_skills(skill_text)

                st.write("✅ Extracted Skills:", ", ".join(sorted(extracted)))

                role_match = df_rs[df_rs['position_title'].str.lower().str.contains(desired_role.lower())]
                if not role_match.empty:
                    ref_skills_text = role_match.iloc[0]['Required Skills']
                    ref_skills = set(s.strip().lower() for s in re.split(r'[.,\n\-\u2022|]', str(ref_skills_text)) if len(s.strip()) > 1)
                    matched = {s for s in extracted if any(re.search(rf"\b{s}\b", r) for r in ref_skills)}
                    missing = ref_skills - matched

                    resume_score = round((len(matched) / len(ref_skills)) * 100, 2) if ref_skills else 0
                    st.metric(label="📊 Resume Match Score", value=f"{resume_score}%")
                    st.write("✅ Matched Skills:", matched)
                    st.write("❌ Missing Skills:", missing)

                    if st.toggle("✨ Show Profile Richness Analysis"):
                        profile_df = compute_profile_features(parsed, matched)
                        profile_score = ll_model.predict(profile_df.values)[0]
                        st.write(f"🔎 Predicted Tier: **{profile_score}**")
                        st.dataframe(profile_df)

                    if len(missing) > len(matched):
                        st.subheader("🎓 Learning Recommendations")
                        df_cc['match'] = df_cc['Skills'].apply(lambda x: any(ms in str(x).lower() for ms in missing))
                        st.write("Courses from Coursera:")
                        st.dataframe(df_cc[df_cc['match']][['title', 'Skills', 'URL']].head(3))

                        g_courses = model.generate_content(f"Suggest 5 online courses from any platform to help someone learn the following skills: {', '.join(missing)}.").text
                        st.write("🌐 Gemini Course Suggestions:")
                        st.markdown(g_courses)

                        pros = df_ll[df_ll['Current_Role'].str.contains(desired_role.split()[0], case=False, na=False)][['Full_Name', 'Contact_mail']].dropna().head(3)
                        st.write("👥 Contact These Professionals:")
                        st.dataframe(pros)

                        st.markdown(model.generate_content(f"{name} wants to become a {desired_role} but is missing key skills like: {', '.join(missing)}. Suggest a short message asking professionals for career growth advice.").text)
                    else:
                        job_row = df_jd[df_jd['Role'].str.contains(desired_role, case=False, na=False)].iloc[0]
                        referral = model.generate_content(f"{name} is applying for the role of {job_row['Job Title']} at {job_row['Company']}. They have matching skills: {', '.join(matched)}. Write a professional referral request email.").text
                        st.subheader("📨 Referral Email:")
                        st.markdown(referral)

                        st.subheader("📍 Additional Job Listings")
                        st.dataframe(df_jd[df_jd['Role'].str.contains(desired_role, case=False, na=False)][['Job Title', 'Company', 'Country']].head(3))

                        st.subheader("👥 Recruiters to Contact")
                        st.dataframe(df_ll[df_ll['Current_Role'].str.contains(desired_role.split()[0], case=False, na=False)][['Full_Name', 'Contact_mail']].dropna().head(3))
