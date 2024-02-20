# cicd-practice

Docker 및 GitHub Action을 활용한 CICD 실습을 위한 프로젝트.

## 사전 준비사항

- YAML과 Docker를 이해하고 있으면 도움이 되요. 잘 모른다면 아래 자료를 한 번 읽어봐주세요!
  - YAML이란 - jnine.log
  - [왕초보를 위한 도커 사용법](https://mysetting.io/slides/xxj85vnvey)
- 아래 개발환경이 세팅되어 있어야 해요.
  - Docker Desktop
  - 코드를 편집할 수 있는 에디터
- python 3.10.12

## 개발 환경 실행

> 필수 패키지 설치

```bash
pip install -r requirements.txt
```

> 실행

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Docker 빌드 및 배포

> Docker 이미지 빌드

```bash
docker build -t jaebum26/cicd([이미지명]:[태그]) -f Dockerfile .
```

> Docker 로컬 실행

```bash
docker run -it --rm -p 8000:8000 [이미지명]:[태그]
```

> Docker 이미지 업로드

```bash
docker push [이미지명]:[태그]
```