$ErrorActionPreference = "Stop"

# 프로젝트 루트 기준 실행

Write-Host "== Lambda zip build 시작 =="

# 1) 기존 산출물 삭제
Remove-Item .\build -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .\beach-ai.zip -Force -ErrorAction SilentlyContinue

# 2) 빌드 폴더 생성
New-Item -ItemType Directory -Path .\build | Out-Null

# 3) Lambda용 의존성 설치
# Python 3.13 + x86_64 기준
py -m pip install `
  --target .\build `
  --platform manylinux2014_x86_64 `
  --implementation cp `
  --python-version 3.13 `
  --only-binary=:all: `
  --upgrade `
  -r .\requirements.txt

# 4) 앱 소스 복사
Copy-Item .\main.py .\build\
Copy-Item .\app .\build\app -Recurse
Copy-Item .\beaches.json .\build\

# 필요하면 같이 복사
if (Test-Path .\README.md) {
    Copy-Item .\README.md .\build\
}

# 5) build 폴더 내부를 zip으로 압축
Set-Location .\build
Compress-Archive -Path * -DestinationPath ..\beach-ai.zip -Force
Set-Location ..

Write-Host "== 완료: beach-ai.zip 생성됨 =="
