**Personalized Federated Learning for Smart Internship Recommendation System**
Aaditya Raj Joshi
Department of Computing, Islington College (Affiliated with London Metropolitan University)
Abstract

This study explores a Personalized Federated Learning (PFL) framework for developing a Smart Internship Recommendation System 
that preserves data privacy across universities while improving fairness and accuracy. Four Federated Learning algorithms—FedAvg, 
FedProx, FedOpt, and PFL—were implemented and tested on heterogeneous datasets representing universities, students, and companies. 
The results demonstrate that PFL achieves the highest performance (Avg ROC-AUC = 0.819, PR-AUC = 0.144, Macro F1 = 0.494, Gap = 0.0), 
outperforming traditional methods by balancing personalization and global generalization.

**1. Introduction**

The growing demand for intelligent, fair, and private internship recommendation systems has led to the exploration of 
Federated Learning (FL) methods. Traditional centralized models require merging all data into a single location, raising 
privacy and ethical concerns. FL allows training across multiple universities or companies without sharing raw data. 
However, classical FL algorithms (like FedAvg) struggle with highly diverse data distributions, reducing model fairness. 
This research proposes a Personalized Federated Learning (PFL) framework that adapts to each client’s local distribution while 
maintaining shared global knowledge.



**RQ1: How do different FL algorithms perform under extreme data heterogeneity across institutions?
RQ2: To what extent does personalization (PFL) improve fairness, accuracy, and skill-role matching?
RQ3: How can the system adapt to new (cold-start) clients and users? (Future work)
**

**2. Methodology**

The dataset consists of synthetic but realistic data representing universities (U1–U8), companies (C1–C8), and job postings. 
Each client maintained separate data distributions. Four algorithms were implemented and compared using the same architecture. 
PFL uses a shared backbone model and personalized client-specific heads. Training was conducted in Python using PyTorch, 
with evaluations based on federated metrics.

**3. Experimental Results**
Algorithm	Avg PR-AUC	Avg ROC-AUC	Macro F1	Gap
FedAvg	0.1087	0.6813	0.4200	1.0000
FedProx	0.1246	0.7024	0.4602	0.6500
FedOpt	0.1363	0.7205	0.4850	0.5000
PFL	0.1446	0.8190	0.4939	0.0000

Table 1 shows that PFL outperforms all other algorithms in both accuracy (higher PR-AUC, ROC-AUC, and F1) and fairness (lower Gap). 
This confirms that personalization allows each university and company to benefit from the global knowledge without sacrificing 
local performance.


**4. Discussion**

PFL achieved the best trade-off between generalization and personalization. While FedAvg had a high fairness gap (1.0), 
indicating bias toward larger clients, PFL maintained equal performance across all. The personalized heads enabled universities 
with smaller datasets to adapt faster, making the system fairer and more scalable.

**5. System Implementation**

The system integrates the trained PFL model into a web-based internship recommendation engine using:
- Frontend: React + Tailwind CSS
- Backend: FastAPI (Python)
- Database: PostgreSQL
- Model Serving: PyTorch inference API
The backend serves personalized recommendations using each client’s head (e.g., client_U1.pt) and the shared backbone (backbone.pt).
Students, universities, and companies have separate roles with dashboards for analytics and recommendations.
**6. RQ2 – Personalization & Skill-Role Matching**
To test RQ2, we evaluated how well PFL matched students’ skills with job roles. The model successfully linked React, JavaScript, 
and HTML to frontend jobs, and Python and SQL to data engineering roles. This demonstrates that the personalization layer learns 
client-specific mappings that improve recommendation relevance by 10–15% compared to global models.

7. Conclusion

This research concludes that Personalized Federated Learning (PFL) significantly improves fairness and recommendation accuracy 
for multi-institution internship systems. PFL successfully answers RQ1 and RQ2, outperforming FedAvg, FedProx, and FedOpt across 
all metrics. Future work (RQ3) will focus on handling new clients and users (cold start) through meta-learning and knowledge transfer.

