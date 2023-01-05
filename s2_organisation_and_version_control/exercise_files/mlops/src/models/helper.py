from pathlib import Path

def get_source_dir():
    source_dir = Path(__file__).resolve().parents[1]
    return str(source_dir)

def get_project_dir():
    project_dir = Path(__file__).resolve().parents[2]
    return str(project_dir)



