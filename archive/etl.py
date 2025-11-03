import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import os

def extract_data():
    """Load raw CSV files"""
    students = pd.read_csv('./test-data/raw/students.csv')
    jobs = pd.read_csv('./test-data/raw/jobs.csv')
    interactions = pd.read_csv('./test-data/raw/interactions.csv')
    companies = pd.read_csv('./test-data/raw/companies.csv')
    return students, jobs, interactions, companies

def transform_data(students, jobs, interactions, companies):
    """Clean and transform features"""
    # Handle missing values
    students['GPA'].fillna(students['GPA'].mean(), inplace=True)

    # Normalize numerical features
    scaler = StandardScaler()
    students['GPA_normalized'] = scaler.fit_transform(students[['GPA']])

    # Vectorize text features (skills, job requirements)
    if 'skills' in students.columns:
        tfidf = TfidfVectorizer()
        _ = tfidf.fit_transform(students['skills'])  # vector not stored here, just an example

    # Merge datasets into a single training-ready frame
    merged_data = pd.merge(interactions, students, on='user_id')
    merged_data = pd.merge(merged_data, jobs, on='job_id')
    merged_data = pd.merge(merged_data, companies, on='company_id')

    return merged_data

def partition_for_federated(data, partition_column='university'):
    """
    Partition dataset by university to simulate extreme heterogeneity.
    Saves each university's local dataset + generates heterogeneity report.
    """
    os.makedirs('./test-data/processed', exist_ok=True)
    stats = []

    if partition_column not in data.columns:
        raise KeyError(f"Required column '{partition_column}' missing in merged data")

    for client_id, group in data.groupby(partition_column):
        # Save each client dataset
        client_dir = f'./test-data/processed/{partition_column}_client_{client_id}'
        os.makedirs(client_dir, exist_ok=True)
        group.to_csv(f'{client_dir}/data.csv', index=False)

        # Track heterogeneity stats
        client_stats = {
            'client_id': client_id,
            'samples': len(group),
            'avg_gpa': group['GPA'].mean() if 'GPA' in group.columns else None,
            'gpa_std': group['GPA'].std() if 'GPA' in group.columns else None,
            'sex_distribution': group['sex'].value_counts(normalize=True).to_dict()
                              if 'sex' in group.columns else None,
            'major_diversity': group['major'].nunique() if 'major' in group.columns else None,
            'role_diversity': group['role'].nunique() if 'role' in group.columns else None,
            'industry_diversity': group['industry'].nunique() if 'industry' in group.columns else None
        }
        stats.append(client_stats)

    # Save heterogeneity report
    pd.DataFrame(stats).to_csv(
        f'./test-data/processed/{partition_column}_heterogeneity_report.csv', index=False
    )
    print(f"✅ Created {len(stats)} {partition_column}-based client partitions")
    return stats


def main():
    # ETL Pipeline
    print("Extracting data...")
    students, jobs, interactions, companies = extract_data()
    
    print("Transforming data...")
    transformed_data = transform_data(students, jobs, interactions, companies)
    
    print("Partitioning for federated learning...")
    partition_for_federated(transformed_data, partition_column="university")
    
    print("ETL pipeline completed!")

if __name__ == "__main__":
    main()