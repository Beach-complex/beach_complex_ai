import os
from dataclasses import dataclass
from pathlib import Path


def load_env_file(path: Path = Path(__file__).parents[1] / ".env") -> None:
    """Load local .env values without overriding real environment variables."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


load_env_file()


@dataclass(frozen=True)
class AppConfig:
    aws_region: str
    openweather_secret_name: str
    openweather_base_url: str = "https://api.openweathermap.org/data/2.5/weather"

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            aws_region=os.environ["AWS_REGION"],
            openweather_secret_name=os.environ["OPENWEATHER_SECRET_NAME"],
        )


settings = AppConfig.from_env()

AWS_REGION = settings.aws_region
OPENWEATHER_SECRET_NAME = settings.openweather_secret_name
