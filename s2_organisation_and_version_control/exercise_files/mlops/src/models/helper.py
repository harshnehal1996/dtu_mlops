from pathlib import Path

def get_source_dir() -> str:
    source_dir = Path(__file__).resolve().parents[1]
    return str(source_dir)

def get_project_dir() -> str:
    project_dir = Path(__file__).resolve().parents[2]
    return str(project_dir)



