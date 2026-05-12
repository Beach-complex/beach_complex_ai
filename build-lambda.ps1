$ErrorActionPreference = "Stop"

# 프로젝트 루트 기준 실행
$BuildDir = ".\.lambda-build"
$PackagePath = ".\beach-ai.zip"

Write-Host "== Lambda zip build 시작 =="

# 1) 기존 산출물 삭제
if (Test-Path $BuildDir) {
    Remove-Item $BuildDir -Recurse -Force
}
if (Test-Path $PackagePath) {
    Remove-Item $PackagePath -Force
}

# 2) 빌드 폴더 생성
New-Item -ItemType Directory -Path $BuildDir | Out-Null
New-Item -ItemType Directory -Path "$BuildDir\vendor" | Out-Null

# 3) Lambda용 의존성 설치
# Python 3.13 + x86_64 기준
py -m pip install `
  --target "$BuildDir\vendor" `
  --platform manylinux2014_x86_64 `
  --implementation cp `
  --python-version 3.13 `
  --only-binary=:all: `
  --upgrade `
  -r .\requirements.txt

if ($LASTEXITCODE -ne 0) {
    throw "pip install 실패: Lambda 패키지 생성을 중단합니다."
}

# 4) 앱 소스 복사
Copy-Item .\main.py $BuildDir\
Copy-Item .\app "$BuildDir\app" -Recurse
Copy-Item .\beaches.json $BuildDir\

# 필요하면 같이 복사
if (Test-Path .\README.md) {
    Copy-Item .\README.md $BuildDir\
}

# 5) build 폴더 내부를 zip으로 압축
Push-Location $BuildDir
Compress-Archive -Path * -DestinationPath "..\beach-ai.zip" -Force
Pop-Location

Write-Host "== 완료: beach-ai.zip 생성됨 =="
