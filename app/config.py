from pathlib import Path

from .settings import get_settings

_SETTINGS = get_settings()

DATA_DIR = (
    _SETTINGS.data_dir
    if _SETTINGS.data_dir.is_absolute()
    else Path(__file__).resolve().parents[1] / _SETTINGS.data_dir
)
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
SCRAPE_DIR = DATA_DIR / "raw"  # reuse raw dir for scraped artifacts

PROC_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Models
EMB_MODEL_NAME = _SETTINGS.emb_model_name
RERANKER_NAME = _SETTINGS.reranker_name
