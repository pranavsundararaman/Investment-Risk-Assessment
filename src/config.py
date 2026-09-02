from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from .env for local development
load_dotenv(PROJECT_ROOT / ".env")

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

NIFTY50_CSV = CONFIG_DIR / "nifty50.csv"


def get_secret(name: str) -> str | None:
    """
    Get a configuration value from Streamlit Secrets when running
    on Streamlit Cloud, otherwise fall back to environment variables.
    """
    try:
        import streamlit as st

        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.getenv(name)


DB_HOST = get_secret("DB_HOST")
DB_PORT = get_secret("DB_PORT")
DB_NAME = get_secret("DB_NAME")
DB_USER = get_secret("DB_USER")
DB_PASSWORD = get_secret("DB_PASSWORD")