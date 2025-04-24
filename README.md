
# Theta Career Assistant – AI-Powered Job Search Tool

## Overview

**Theta Career Assistant** is an AI-powered career guidance platform that helps users analyze their resumes, match their skills with job role requirements, identify learning gaps, and generate professional networking messages. It integrates natural language models, machine learning, and real-world datasets into an interactive application designed to streamline the job search and upskilling process.

This project supports **two user interfaces**:
- **Structured (form-based) interface** to validate backend logic step-by-step.
- **Conversational Streamlit chatbot interface** for a real-time, natural interaction experience.


## Features

- Resume PDF upload and skill extraction
- Match against role-based required skills
- Skill gap analysis and course recommendations from dataset
- Predicts user's professional tier (Beginner/Intermediate/Advanced)
- Generates cold emails and message templates using Gemini API
- Lists recruiters, job openings, and top hiring locations


## Directory Structure

```
cap5771sp25-project/
│
├── README.md                     # This file
├── Reports/
│   └── milestone1_report.pdf
|   └── milestone2_report.pdf 
|   └── milestone3_report.pdf    # Final report with methods and insights
│
├── Scripts/                      # Jupyter notebooks (Milestone 1 & 2 logic)
│   ├── LinkedIn.ipynb
│   ├── Coursera_details.ipynb
│   ├── Role_based_skills.ipynb
│   └── job_data.ipynb
│
├── Streamlit/                    # Streamlit chatbot logic (Milestone 3)
│   └── visual.py
│
├── assets/
│   ├── models/                   # Cleaned CSVs and trained models (.pkl)
│   │   ├── Coursera
│   │   ├── Job_descriptions
│   │   ├── LinkedIn
│   │   └── Roles_based_skills
│   ├── Sample_resume.pdf         # Sample resume for testing
│   └── Output/                   # UI output screenshots and result PDFs
```
## Technologies Used

- **Python 3.8+**
- **Streamlit** – For building the chatbot interface
- **Google Generative AI (Gemini)** – For skill suggestions, messaging, and course recommendations
- **pdfplumber** – To extract text from uploaded resumes
- **pandas**, **scikit-learn**, **joblib** – Data preprocessing and ML modeling
- **Jupyter Notebooks** – For prototyping and data analysis


## Datasets & Models

| Dataset              | Usage                                                 |
|----------------------|-------------------------------------------------------|
| `Coursera`           | Course titles, skills, ratings → for upskilling advice |
| `LinkedIn`           | Simulated user profiles and recruiters → for networking |
| `Role-Based Skills`  | Required skills by job title → for resume-role match  |
| `Job Descriptions`   | Titles, salary, locations → for job listing insights  |

Each dataset was cleaned and saved in `.csv` format. Corresponding ML models were trained and serialized as `.pkl` files.


## Model Functions

- **Classification Models**:
  - Predict `Experience_Binary`, `Profile_Tier`, and `Popularity_Class`
  - Algorithms: Logistic Regression, Random Forest

- **Regression Model**:
  - Predicts average salary from job features
  - Algorithm: Random Forest Regressor

- **Resume Scoring**:
  - Computes skill match % against job requirements
  - Extracts matched/missing skills
  - Calculates feature vector for profile richness


## How to Use

1. **Run the Chatbot**:
   ```bash
   streamlit run Streamlit/visual.py
   ```

2. **Follow the Prompts**:
   - Enter your name, current role, and target role
   - View required skills and hiring locations
   - Upload your resume (PDF)
   - Review skill match, course recommendations, and recruiter contacts

3. **View Outputs**:
   - Cold email templates
   - Coursera courses (from dataset and Gemini)
   - Role-based resume score
   - Predicted career tier


## Sample Outputs

- `assets/Output/normal_output.pdf` – Structured UI run
- `assets/Output/output.pdf` – Streamlit Chatbot UI run


## Final Report

The comprehensive summary of design, methodology, modeling, evaluation, and screenshots is provided in:  
`Reports/milestone3_report.pdf`
Link for PPT: https://drive.google.com/drive/u/2/folders/1Vn-mqMNqLcvMJ1ozCFwWG4FlccUSdsyj 


## Author

**Swetha Gendlur Nagarajan**  
Master’s in Applied Data Science  
University of Florida, April 2025

