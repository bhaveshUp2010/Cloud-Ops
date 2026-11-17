from src.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    print(f"GEMINI_API_KEY configured: {'yes' if settings.GEMINI_API_KEY else 'no'}")