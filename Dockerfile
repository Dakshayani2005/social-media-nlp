FROM python:3.9-slim

WORKDIR /app

# 🔹 Install system build dependencies (THIS FIXES gcc ERROR)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 🔹 Upgrade pip
RUN pip install --upgrade pip

# 🔹 Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 🔹 Download spaCy model
RUN python -m spacy download en_core_web_sm

# 🔹 Copy project files
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
