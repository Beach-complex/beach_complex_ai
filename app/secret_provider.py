import json
from functools import lru_cache

import boto3
from botocore.exceptions import ClientError


class OpenWeatherApiKeyError(Exception):
    pass


class OpenWeatherApiKeyProvider:
    def __init__(self, secret_name: str, aws_region: str) -> None:
        self._secret_name = secret_name
        self._client = boto3.client("secretsmanager", region_name=aws_region)

    @lru_cache(maxsize=1)
    def get_api_key(self) -> str:
        if not self._secret_name:
            raise OpenWeatherApiKeyError(
                "OPENWEATHER_SECRET_NAME 환경 변수가 설정되지 않았습니다."
            )

        try:
            response = self._client.get_secret_value(SecretId=self._secret_name)
            secret_string = response.get("SecretString")

            if not secret_string:
                raise OpenWeatherApiKeyError("SecretString 이 비어 있습니다.")

            secret_obj = json.loads(secret_string)
            api_key = secret_obj.get("OPENWEATHER_API_KEY", "")

            if not api_key:
                raise OpenWeatherApiKeyError(
                    "시크릿에 OPENWEATHER_API_KEY 키가 없습니다."
                )

            return api_key

        except ClientError as e:
            raise OpenWeatherApiKeyError(
                f"Secrets Manager 조회 실패: {e.response['Error']['Code']}"
            ) from e
