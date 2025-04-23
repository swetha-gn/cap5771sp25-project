import streamlit as st
import pandas as pd
import joblib
import re
import pdfplumber
import matplotlib.pyplot as plt
import google.generativeai as genai

#  Configure Gemini 
genai.configure(api_key="AIzaSyA3z0ROR3eBYxD1cqaJj0Jw2fTGnc83gZU")
model = genai.GenerativeModel("models/gemini-1.5-flash")

#  Load Data & Models 
df_cc = pd.read_csv("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Coursera/cleaned_coursera_data.csv")
df_jd = pd.read_csv("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Job_descriptions/df_chatbot.csv")
df_ll = pd.read_csv("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Linkedin/preprocessed_linkedin_data.csv")
df_rs = pd.read_csv("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Roles_based_skills/final_role_predictions.csv")
ll_model = joblib.load("/Users/swethagendlurnagarajan/Desktop/cap5771sp25-project/IDS/models/Linkedin/best_profile_tier_model.pkl")

#  Resume Parsing Helpers 
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
    return df

def visualize_required_skills_for_role(role_title):
    matched = df_rs[df_rs['position_title'].str.lower().str.contains(role_title.lower())]
    if matched.empty:
        return None
    required_skills_text = matched.iloc[0]['Required Skills']
    skills_list = [s.strip().lower() for s in re.split(r'[.,\n\-\u2022|]', str(required_skills_text)) if len(s.strip()) > 1]
    skill_freq = pd.Series(skills_list).value_counts().head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(skill_freq.index[::-1], skill_freq.values[::-1])
    plt.title(f"Top Required Skills for '{role_title}'")
    plt.xlabel("Frequency")
    plt.tight_layout()
    st.pyplot(plt)

#  Chat Interface 
st.set_page_config(page_title="Gemini Career Chatbot", layout="centered")
st.title("💬 Gemini Career Assistant (Chat Mode)")

if "stage" not in st.session_state:
    st.session_state.stage = "intro"
    st.session_state.data = {}
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def add_user_input_and_advance(prompt, next_stage):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.stage = next_stage
    st.rerun()

if st.session_state.stage == "intro":
    if prompt := st.chat_input("👋 What's your name?"):
        st.session_state.data["name"] = prompt
        add_user_input_and_advance(prompt, "current_role")

elif st.session_state.stage == "current_role":
    if prompt := st.chat_input("💼 What's your current role?"):
        st.session_state.data["current_role"] = prompt
        add_user_input_and_advance(prompt, "desired_role")

elif st.session_state.stage == "desired_role":
    if prompt := st.chat_input("🎯 What role are you aiming for?"):
        st.session_state.data["desired_role"] = prompt
        add_user_input_and_advance(prompt, "locations")

elif st.session_state.stage == "locations":
    if prompt := st.chat_input("📍 See top locations for this role? (yes/no)"):
        add_user_input_and_advance(prompt, "skills_needed")
        if prompt.lower() == "yes":
            top_locs = df_jd[df_jd['Role'].str.contains(st.session_state.data['desired_role'], case=False, na=False)]['Country'].value_counts().head(3)
            top_loc_text = "\n".join([f"{k}: {v}" for k, v in top_locs.items()])
            st.session_state.messages.append({"role": "assistant", "content": f"🌍 Top Locations for this Role:\n{top_loc_text}"})
        st.rerun()

elif st.session_state.stage == "skills_needed":
    role = st.session_state.data['desired_role']
    skill_summary = model.generate_content(f"List 10 key technical and soft skills needed for the role: {role}. Pretend it was extracted from a real job dataset.").text
    st.session_state.messages.append({"role": "assistant", "content": f"📜 Skills needed for this role:\n{skill_summary}"})
    with st.chat_message("assistant"):
        visualize_required_skills_for_role(role)
    curr = st.session_state.data["current_role"]
    st.session_state.stage = "experience" if curr.lower() == role.lower() else "exit_student"
    st.rerun()
#  Continuing the Streamlit Chat Flow 

elif st.session_state.stage == "experience":
    if prompt := st.chat_input("🧠 Do you have experience in this role? (yes/no)"):
        add_user_input_and_advance(prompt, "upload" if prompt.lower() == "yes" else "exit_student")

elif st.session_state.stage == "exit_student":
    role = st.session_state.data["desired_role"]
    name = st.session_state.data["name"]
    df_cc['combined'] = df_cc['title'] + ' ' + df_cc['Skills']
    courses = df_cc[df_cc['combined'].str.contains(role.split()[0], case=False, na=False)][['title', 'URL']].head(3)
    links = "\n".join([f"[{row['title']}]({row['URL']})" for _, row in courses.iterrows()])
    msg = model.generate_content(f"{name} is new to the {role} field. Suggest a short message asking professionals for guidance and learning resources.").text
    st.chat_message("assistant").markdown(f"🎓 Suggested Courses:\n{links}\n\n📨 Cold Email Template:\n{msg}")
    st.stop()

elif st.session_state.stage == "upload":
    uploaded = st.file_uploader("📄 Upload your resume (PDF)", type=["pdf"])
    if uploaded:
        text = extract_text_from_pdf_bytes(uploaded)
        parsed = extract_sections(text)
        skill_text = parsed.get("skills", "") or text
        extracted = extract_skills(skill_text)
        role = st.session_state.data["desired_role"]
        matched = df_rs[df_rs['position_title'].str.lower().str.contains(role.lower())]
        if matched.empty:
            st.error("❌ No matching role found.")
            st.stop()
        ref_skills = set(s.strip().lower() for s in re.split(r'[.,\n\-\u2022|]', matched.iloc[0]['Required Skills']) if len(s.strip()) > 1)
        matched_skills = {s for s in extracted if any(re.search(rf"\\b{s}\\b", r) for r in ref_skills)}
        missing_skills = ref_skills - matched_skills
        score = round((len(matched_skills) / len(ref_skills)) * 100, 2)
        st.chat_message("assistant").markdown(f"📊 Resume Score: **{score}%**\n✅ Matched: {', '.join(matched_skills)}\n❌ Missing: {', '.join(missing_skills)}")

        if st.toggle("✨ Profile Richness Analysis"):
            features = compute_profile_features(parsed, matched_skills)
            predicted_tier = ll_model.predict(features.values)[0]
            st.chat_message("assistant").markdown(f"🔎 Profile Tier: **{predicted_tier}**")
            st.dataframe(features)

        st.session_state.data.update({
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "predicted_tier": predicted_tier if 'predicted_tier' in locals() else "beginner"
        })
        st.session_state.stage = "contact"
        st.rerun()

elif st.session_state.stage == "contact":
    role = st.session_state.data["desired_role"]
    name = st.session_state.data["name"]
    current = st.session_state.data["current_role"]
    tier = st.session_state.data["predicted_tier"]
    matched_skills = st.session_state.data["matched_skills"]
    missing_skills = st.session_state.data["missing_skills"]

    pros = df_ll[df_ll['Current_Role'].str.contains(role.split()[0], case=False, na=False)][['Full_Name', 'Contact_mail']].dropna().head(3)
    st.chat_message("assistant").markdown("👥 Professionals to connect with:")
    st.dataframe(pros)

    if current.lower() != role.lower():
        prompt = f"{name} is transitioning from a {current} to a {role} role. Write a polite cold email asking for guidance and resources."
    elif tier.lower() in ["beginner", "intermediate"]:
        prompt = f"{name} is working as a {current} and wants to grow in the same {role} field. Write a cold email asking for profile feedback and tips."
    else:
        prompt = f"{name} is a {current} with strong skills applying for a {role} role. Write a professional referral request email."

    cold_email = model.generate_content(prompt).text
    st.chat_message("assistant").markdown(f"📨 Cold Email Template:\n\n{cold_email}")
    st.session_state.stage = "end"

elif st.session_state.stage == "end":
    st.chat_message("assistant").markdown("✅ Thank you for using the Gemini Career Assistant!")
