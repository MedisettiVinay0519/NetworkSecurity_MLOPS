# 🔐 Network Security ML Pipeline with FastAPI & Docker

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow)](https://mlflow.org/)
[![DagsHub](https://img.shields.io/badge/DagsHub-231F20?logo=dagshub)](https://dagshub.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-47A248?logo=mongodb)](https://www.mongodb.com/)
[![GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=github-actions)](https://github.com/features/actions)

An **end-to-end MLOps project** for network threat detection. [cite_start]This pipeline automates the journey from raw data ingestion to a containerized production-ready API. [cite: 1, 27]

---

## 🛠️ Tech Stack

<p align="left">
  <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="42"/>
  <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/fastapi/fastapi-original.svg" width="42"/>
  <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/docker/docker-original.svg" width="42"/>
  <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/mongodb/mongodb-original.svg" width="42"/>
  <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/scikitlearn/scikitlearn-original.svg" width="42"/>
  <img src="https://avatars.githubusercontent.com/u/73596471?s=200&v=4" width="42" title="DagsHub"/>
</p>

- [cite_start]**Core:** Python 3.10, FastAPI, Scikit-learn 
- [cite_start]**Data Engineering:** ETL Pipeline (CSV/S3 to MongoDB Atlas) 
- [cite_start]**MLOps:** MLflow (Experiment Tracking), DagsHub (Remote Backend) 
- [cite_start]**DevOps:** Docker, GitHub Actions (CI for Docker builds) 

---

## 🏗️ Project Architecture

### 🔄 ETL Pipeline (Extract, Transform, Load)
[cite_start]Raw data is processed and stored in a NoSQL format before entering the ML pipeline. [cite: 27]
* [cite_start]**Source:** Extracts data from CSV, S3 Buckets, or APIs. [cite: 28, 30, 32]
* [cite_start]**Transformation:** Performs basic cleaning and converts data to JSON. [cite: 36, 37]
* [cite_start]**Load:** Stores the cleaned data into **MongoDB Atlas**. [cite: 38, 39, 48]

### 🧪 ML Pipeline Components
[cite_start]The system follows a modular component-based structure: 
1. [cite_start]**Data Ingestion:** Retrieves data from MongoDB and splits it into train/test sets. [cite: 58, 76]
2. [cite_start]**Data Validation:** Checks schema and identifies **Data Drift** (Distribution shifts). [cite: 85, 96, 140]
3. [cite_start]**Data Transformation:** Implements **KNN Imputer** for missing value handling and scales features. [cite: 142, 197]
4. [cite_start]**Model Training:** Utilizes a **Model Factory** to find the best performing model. [cite: 204, 250]

---

## 🐳 CI/CD & Docker
* [cite_start]**Continuous Integration:** Every push to `main` triggers **GitHub Actions** to build the Docker image. 
* [cite_start]**Containerization:** The entire application is packaged into a **Docker Image** for consistent deployment. [cite: 272, 275]
* **Run Locally:**
  ```bash
  docker build -t network-security-app .
  docker run -p 8000:8000 network-security-app
```
