

import streamlit as st
import pandas as pd
import joblib
import re
import pdfplumber
import google.generativeai as genai

#  Configure Gemini 
genai.configure(api_key="AIzaSyDTo5J7P130C5J7P130C5bUbqwcUADHtb6MXj1_2ms")
model = genai.GenerativeModel("models/gemini-1.5-flash")

#  Load Models & Data 
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

#  Streamlit App Logic 
st.set_page_config(page_title="Gemini Career Chatbot", layout="centered")
st.title("💬 Gemini Career Assistant (Conversational Flow)")

if "stage" not in st.session_state:
    st.session_state.stage = "intro"
    st.session_state.user_data = {}
    st.session_state.messages = []

# Chat handler
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.stage == "intro":
    if prompt := st.chat_input("👋 What's your name?"):
        st.session_state.user_data["name"] = prompt
        st.session_state.stage = "current_role"
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

elif st.session_state.stage == "current_role":
    if prompt := st.chat_input("💼 What's your current role?"):
        st.session_state.user_data["current_role"] = prompt
        st.session_state.stage = "desired_role"
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

elif st.session_state.stage == "desired_role":
    if prompt := st.chat_input("🎯 What role are you aiming for?"):
        st.session_state.user_data["desired_role"] = prompt
        st.session_state.stage = "locations"
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

elif st.session_state.stage == "locations":
    if prompt := st.chat_input("📍 Would you like to see top locations hiring for this role? (yes/no)"):
        st.session_state.stage = "skills_needed"
        st.session_state.messages.append({"role": "user", "content": prompt})
        if prompt.lower() == "yes":
            top_locs = df_jd[df_jd['Role'].str.contains(st.session_state.user_data['desired_role'], case=False, na=False)]['Country'].value_counts().head(3)
            loc_text = "\n".join([f"{k}: {v}" for k, v in top_locs.items()])
            st.session_state.messages.append({"role": "assistant", "content": f"🌍 Top Locations for this Role:\n{loc_text}"})
        st.rerun()

elif st.session_state.stage == "skills_needed":
    role = st.session_state.user_data['desired_role']
    skills = model.generate_content(f"List 10 key technical and soft skills needed for the role: {role}. Pretend it was extracted from a real job dataset.").text
    st.session_state.messages.append({"role": "assistant", "content": f"📜 Here are common skills needed for this role:\n{skills}"})
    if st.session_state.user_data['current_role'].lower() in ["student", "intern"] or st.session_state.user_data['current_role'].lower() != role.lower():
        st.session_state.stage = "exit_student"
    else:
        st.session_state.stage = "experience"
    st.rerun()

elif st.session_state.stage == "experience":
    if prompt := st.chat_input("🧠 Do you have experience in this role? (yes/no)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        if prompt.strip().lower() != "yes":
            st.session_state.stage = "exit_student"
        else:
            st.session_state.stage = "upload"
        st.rerun()

elif st.session_state.stage == "exit_student":
    role = st.session_state.user_data['desired_role']
    df_cc['combined'] = df_cc['title'] + ' ' + df_cc['Skills']
    courses = df_cc[df_cc['combined'].str.contains(role.split()[0], case=False, na=False)][['title', 'URL']].head(3)
    links = "\n".join([f"[{row['title']}]({row['URL']})" for _, row in courses.iterrows()])
    msg = model.generate_content(f"{st.session_state.user_data['name']} is new to the {role} field. Suggest a short message asking professionals for guidance and learning resources.").text
    st.chat_message("assistant").markdown(f"🎓 Suggested Coursera Courses:\n{links}\n\n📨 Message Template:\n{msg}")
    st.stop()

elif st.session_state.stage == "upload":
    uploaded = st.file_uploader("📄 Please upload your resume (PDF)", type=["pdf"])
    if uploaded:
        text = extract_text_from_pdf_bytes(uploaded)
        parsed = extract_sections(text)
        skills = extract_skills(parsed.get("skills", "") or text)
        role = st.session_state.user_data['desired_role']
        matched = df_rs[df_rs['position_title'].str.lower().str.contains(role.lower())]
        if matched.empty:
            st.error("❌ No matching role found in database.")
            st.stop()
        ref_skills = set(s.strip().lower() for s in re.split(r'[.,\n\-\u2022|]', matched.iloc[0]['Required Skills']) if len(s.strip()) > 1)
        matched_skills = {s for s in skills if any(re.search(rf"\b{s}\b", r) for r in ref_skills)}
        missing_skills = ref_skills - matched_skills
        score = round((len(matched_skills) / len(ref_skills)) * 100, 2) if ref_skills else 0
        msg = f"📊 Resume Score: **{score}%**\n✅ Matched: {', '.join(sorted(matched_skills))}\n❌ Missing: {', '.join(sorted(missing_skills))}"
        st.chat_message("assistant").markdown(msg)

        if st.toggle("✨ Run Profile Richness Analysis"):
            feat = compute_profile_features(parsed, matched_skills)
            tier = ll_model.predict(feat.values)[0]
            st.markdown(f"🔎 Profile Tier: **{tier}**")
            st.dataframe(feat)

        if len(missing_skills) > len(matched_skills):
            df_cc['match'] = df_cc['Skills'].apply(lambda x: any(ms in str(x).lower() for ms in missing_skills))
            recs = df_cc[df_cc['match']][['title', 'URL']].head(3)
            st.write("🎓 Courses Based on Missing Skills:")
            st.dataframe(recs)
            g_suggest = model.generate_content(f"Suggest online courses for: {', '.join(missing_skills)}").text
            st.write("🎓 Gemini Suggestions:")
            st.markdown(g_suggest)
            st.session_state.stage = "contact"
        else:
            job_row = df_jd[df_jd['Role'].str.contains(role, case=False, na=False)].iloc[0]
            mail = model.generate_content(f"{st.session_state.user_data['name']} is applying for the role of {job_row['Job Title']} at {job_row['Company']}. They have matching skills: {', '.join(matched_skills)}. Write a professional referral request email.").text
            st.write("📨 Referral Email:")
            st.markdown(mail)
            st.session_state.stage = "contact"
        st.rerun()

elif st.session_state.stage == "contact":
    role = st.session_state.user_data['desired_role']
    pros = df_ll[df_ll['Current_Role'].str.contains(role.split()[0], case=False, na=False)][['Full_Name', 'Contact_mail']].dropna().head(3)
    st.write("👥 Professionals to Connect With:")
    st.dataframe(pros)
