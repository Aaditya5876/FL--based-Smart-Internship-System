import pandas as pd
import numpy as np
import random
from faker import Faker
import json

fake = Faker()

# -----------------
# Parameters
# -----------------
num_students = 1000
num_companies = 100
num_jobs = 500
num_interactions = 5000

# -----------------
# Companies
# -----------------
company_ids = [f"C{str(i).zfill(4)}" for i in range(1, num_companies+1)]
companies = []
for cid in company_ids:
    companies.append({
        "company_id": cid,
        "company_name": fake.company(),
        "industry": fake.job(),
        "company_size": random.choice(["Small", "Medium", "Large"]),
        "country": fake.country(),
        "state": fake.state()
    })
df_companies = pd.DataFrame(companies)

# -----------------
# Jobs
# -----------------
job_ids = [f"J{str(i).zfill(5)}" for i in range(1, num_jobs+1)]
jobs = []
skills_pool = ["Python","Java","C++","SQL","Data Analysis","Machine Learning",
               "Communication","Management","Design","AI","Cloud Computing","Marketing"]
for jid in job_ids:
    company_id = random.choice(company_ids)
    jobs.append({
        "job_id": jid,
        "company_id": company_id,
        "title": fake.job(),
        "role": fake.job(),
        "location": fake.city(),
        "work_type": random.choice(["Internship","Full-time","Part-time"]),
        "salary_min": random.randint(30000, 70000),
        "salary_max": random.randint(70001, 150000),
        "posting_date": fake.date_this_year(),
        "skills_required": ",".join(random.sample(skills_pool, k=random.randint(2,5)))
    })
df_jobs = pd.DataFrame(jobs)

# -----------------
# Students
# -----------------
student_ids = [f"U{str(i).zfill(5)}" for i in range(1, num_students+1)]
majors_pool = ["Computer Science","Business","Biology","Economics","Physics",
               "Engineering","Psychology","Marketing"]
students = []
for uid in student_ids:
    students.append({
        "user_id": uid,
        "name": fake.name(),
        "sex": random.choice(["Male","Female"]),
        "age": random.randint(20,30),
        "major": random.choice(majors_pool),
        "GPA": round(random.uniform(2.0,4.0),2),
        "location": fake.city(),
        "university": fake.company_suffix(),
        "skills": ",".join(random.sample(skills_pool, k=random.randint(2,5)))
    })
df_students = pd.DataFrame(students)

# -----------------
# Interactions (applications)
# -----------------
interactions = []
for _ in range(num_interactions):
    student_id = random.choice(student_ids)
    job_id = random.choice(job_ids)
    student_skills = set(df_students.loc[df_students.user_id==student_id,"skills"].values[0].split(","))
    job_skills = set(df_jobs.loc[df_jobs.job_id==job_id,"skills_required"].values[0].split(","))
    match_score = len(student_skills & job_skills)/len(student_skills | job_skills)
    recommended = 1 if match_score > 0.5 else 0
    interactions.append({
        "user_id": student_id,
        "job_id": job_id,
        "match_score": match_score,
        "recommended": recommended
    })
df_interactions = pd.DataFrame(interactions)

# -----------------
# Save CSV / JSON
# -----------------
df_companies.to_csv("./raw/companies.csv", index=False)
df_jobs.to_csv("./raw/jobs.csv", index=False)
df_students.to_csv("./raw/students.csv", index=False)
df_interactions.to_csv("./raw/interactions.csv", index=False)

# # Optional: save students JSON for extra
# students_json = df_students.to_dict(orient="records")
# with open("students.json","w") as f:
#     json.dump(students_json, f, indent=4)
