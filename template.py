from pathlib import Path

folders = [
    "src",
    "documents",
    "templates",
    "static",
    "uploads",
    "data"
]

files = [
    "app.py",
    "main.py",
    "data_ingestion.py",
    "Dockerfile",
    "requirements.py",
    ".env",


    "src/__init__.py",
    "src/config.py",
    "src/db.py",
    "src/ingestion.py",
    "src/models.py",
    "src/self_rag.py",
    "src/vectorstore.py",

    "templates/index.html",
    "templates/styles.css",
    "templates/app.js",
]

def main():
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

    for file in files:
        Path(file).touch(exist_ok=True)
    print("Folder Structre Created Successfully")

if __name__ == "__main__":
    main()