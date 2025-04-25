import streamlit as st
import pandas as pd
import joblib
import re
import pdfplumber
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key="AIzaSyA3z0ROR3eBYxD1cqaJj0Jw2fTGnc83gZU")
model = genai.GenerativeModel("models/gemini-1.5-flash")

# Load Models & Data
cc_model = joblib.load("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Coursera/best_course_popularity_model.pkl")
df_cc = pd.read_csv("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Coursera/cleaned_coursera_data.csv")

jdc_model = joblib.load("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Job_descriptions/random_forest_classifier.pkl")
jdr_model = joblib.load("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Job_descriptions/random_forest_regressor.pkl")
df_jd = pd.read_csv("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Job_descriptions/df_chatbot.csv")

ll_model = joblib.load("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Linkedin/best_profile_tier_model.pkl")
df_ll = pd.read_csv("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Linkedin/preprocessed_linkedin_data.csv")

rs_model = joblib.load("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Roles_based_skills/best_model.joblib")
df_rs = pd.read_csv("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Roles_based_skills/final_role_predictions.csv")

# Resume Parsing Helpers
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

# Streamlit Chat UI
st.set_page_config(page_title="Theta Career Chatbot", layout="centered")
st.title("💬 Theta Career Assistant (Chat Style UI)")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.stage = "ask_name"
    st.session_state.user_data = {}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def bot_reply(text):
    st.session_state.messages.append({"role": "assistant", "content": text})
    with st.chat_message("assistant"):
        st.markdown(text)

def user_reply(text):
    st.session_state.messages.append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.markdown(text)

if prompt := st.chat_input("Type your response..."):
    user_reply(prompt)

    if st.session_state.stage == "ask_name":
        st.session_state.user_data["name"] = prompt
        st.session_state.stage = "current_role"
        bot_reply("💼 What's your current role?")

    elif st.session_state.stage == "current_role":
        st.session_state.user_data["current_role"] = prompt
        st.session_state.stage = "desired_role"
        bot_reply("🎯 What role are you aiming for?")

    elif st.session_state.stage == "desired_role":
        st.session_state.user_data["desired_role"] = prompt
        st.session_state.stage = "location_check"
        bot_reply("📍 Would you like to see top locations hiring for this role? (yes/no)")

    elif st.session_state.stage == "location_check":
        if prompt.strip().lower() == "yes":
            top_locs = df_jd[df_jd['Role'].str.contains(st.session_state.user_data['desired_role'], case=False, na=False)]['Country'].value_counts().head(3)
            bot_reply("🌍 **Top Hiring Locations:**\n" + top_locs.to_frame().to_markdown())
            st.session_state.stage = "job_summary_check"
            bot_reply("📝 Would you like to see available job summaries for this role, including sector and salary range? (yes/no)")

    elif st.session_state.stage == "job_summary_check":
        if prompt.strip().lower() == "yes":
            role = st.session_state.user_data['desired_role']
            jobs = df_jd[df_jd['Role'].str.contains(role, case=False, na=False)][['Role', 'Company', 'Sector', 'Salary Starts From', 'Salary To']].dropna().head(5)
            jobs['Salary Range'] = jobs['Salary Starts From'].astype(int).astype(str) + ' - ' + jobs['Salary To'].astype(int).astype(str)
            jobs_display = jobs[['Role', 'Company', 'Sector', 'Salary Range']]
            bot_reply("💼 **Available Job Listings:**\n" + jobs_display.to_markdown(index=False))
        st.session_state.stage = "skills_needed"

        role = st.session_state.user_data['desired_role']
        skills = model.generate_content(f"List 10 key technical and soft skills needed for the role: {role}.").text
        bot_reply("📜 **Common Skills for This Role:**\n" + skills)

        if st.session_state.user_data['current_role'].lower() in ['student', 'intern'] or st.session_state.user_data['current_role'].lower() != role.lower():
            st.session_state.stage = "early_exit"
            df_cc['combined'] = df_cc['title'] + ' ' + df_cc['Skills']
            courses = df_cc[df_cc['combined'].str.contains(role.split()[0], case=False, na=False)][['title', 'URL']].head(3)
            course_links = "\n".join([f"[{row['title']}]({row['URL']})" for _, row in courses.iterrows()])
            bot_reply(f"🎓 Recommended Courses:\n{course_links}")
            msg = model.generate_content(f"{st.session_state.user_data['name']} is new to the {role} field. Suggest a short message asking professionals for guidance.").text
            bot_reply("📨 Message Template to Seek Guidance:\n" + msg)
        else:
            st.session_state.stage = "experience"
            bot_reply("🧠 Do you have experience in this role? (yes/no)")

    elif st.session_state.stage == "experience":
        if prompt.strip().lower() != "yes":
            st.session_state.stage = "early_exit"
            role = st.session_state.user_data['desired_role']
            df_cc['combined'] = df_cc['title'] + ' ' + df_cc['Skills']
            courses = df_cc[df_cc['combined'].str.contains(role.split()[0], case=False, na=False)][['title', 'URL']].head(3)
            course_links = "\n".join([f"[{row['title']}]({row['URL']})" for _, row in courses.iterrows()])
            bot_reply(f"🎓 Beginner-Friendly Courses:\n{course_links}")
            msg = model.generate_content(f"{st.session_state.user_data['name']} is interested in the {role} role but doesn't yet have experience. Suggest a message asking for career advice.").text
            bot_reply("📨 Message Template:\n" + msg)
        else:
            st.session_state.stage = "resume_upload"
            bot_reply("📄 Please upload your resume (PDF) using the file uploader above.")

# Resume uploader shown after appropriate stage
if st.session_state.stage == "resume_upload":
    uploaded = st.file_uploader("Upload Resume", type="pdf")
    if uploaded:
        text = extract_text_from_pdf_bytes(uploaded)
        parsed = extract_sections(text)
        skills = extract_skills(parsed.get("skills", "") or text)
        role = st.session_state.user_data['desired_role']
        matched = df_rs[df_rs['position_title'].str.lower().str.contains(role.lower())]
        if matched.empty:
            bot_reply("❌ No matching role found in database.")
        else:
            ref_skills = set(s.strip().lower() for s in re.split(r'[.,\n\-\u2022|]', matched.iloc[0]['Required Skills']) if len(s.strip()) > 1)
            matched_skills = {s for s in skills if any(re.search(rf"\b{s}\b", r) for r in ref_skills)}
            missing_skills = ref_skills - matched_skills
            score = round((len(matched_skills) / len(ref_skills)) * 100, 2)
            st.divider()
            st.subheader("📊 Resume Evaluation Summary")

            st.metric(label="Resume Match Score (%)", value=score)

            bot_reply(f"✅ **Matched Skills:** {', '.join(sorted(matched_skills))}\n\n❌ **Missing Skills:** {', '.join(sorted(missing_skills))}")

            feat = compute_profile_features(parsed, matched_skills)
            tier = ll_model.predict(feat.values)[0]
            bot_reply(f"🧠 **Profile Tier**: {tier}")
            if len(missing_skills) > len(matched_skills):
                df_cc['match'] = df_cc['Skills'].apply(lambda x: any(ms in str(x).lower() for ms in missing_skills))
                recs = df_cc[df_cc['match']][['title', 'URL']].head(3)
                course_links = "\n".join([f"[{row['title']}]({row['URL']})" for _, row in recs.iterrows()])
                g_courses = model.generate_content(f"Suggest online courses for: {', '.join(missing_skills)}").text
                bot_reply(f"🎓 **Courses Based on Missing Skills:**\n{course_links}\n\n{g_courses}")
            else:
                job_row = df_jd[df_jd['Role'].str.contains(role, case=False, na=False)].iloc[0]
                msg = model.generate_content(f"{st.session_state.user_data['name']} is applying for the role of {job_row['Job Title']} at {job_row['Company']}. They have matching skills: {', '.join(matched_skills)}. Write a professional referral request email.").text
                bot_reply("📨 **Referral Email Template:**\n" + msg)
            pros = df_ll[df_ll['Current_Role'].str.contains(role.split()[0], case=False, na=False)][['Full_Name', 'Current_Role', 'Skills', 'Certifications', 'Experience_Years', 'Contact_mail']].dropna().head(3)

            bot_reply("👥 **Professionals in This Field:**\n" + pros.to_markdown(index=False))
            msg = model.generate_content(f"{st.session_state.user_data['name']} is new to the {role} field. Suggest a short message asking professionals for guidance.").text
            bot_reply("📨 Message Template to Seek Guidance:" + msg)
        st.session_state.stage = "done"
        # General Career Insights
        st.divider()
        st.subheader("🌟 General Career Insights from Our Dataset")

        # Top Roles
        top_roles = df_jd['Role'].value_counts().head(3)
        st.write("**Top 3 In-Demand Roles:**")
        st.markdown("".join([f"- {role}" for role in top_roles.index]))

        # Top Skills
        top_skills = df_cc['Skills'].str.split(',').explode().str.strip().value_counts().head(3)
        st.write("**Top 3 In-Demand Skills:**")
        st.markdown("".join([f"- {skill}" for skill in top_skills.index]))

        # Average Salary if available
        if 'Salary Starts From' in df_jd.columns and 'Salary To' in df_jd.columns:
            avg_salary_start = df_jd['Salary Starts From'].dropna().astype(float).mean()
            avg_salary_to = df_jd['Salary To'].dropna().astype(float).mean()
            st.metric(label="💰 Average Salary Range", value=f"${avg_salary_start:,.0f} - ${avg_salary_to:,.0f}")

