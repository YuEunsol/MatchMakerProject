# 🎯 MatchMaker Project

AI 기반 지능형 매칭 시스템으로, 사용자 프로필과 선호도를 기반으로 최적의 파트너를 추천하는 프로젝트입니다.

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [주요 기능](#-주요-기능)
- [시스템 아키텍처](#-시스템-아키텍처)
- [설치 및 실행](#-설치-및-실행)
- [데이터 구조](#-데이터-구조)
- [매칭 알고리즘](#-매칭-알고리즘)
- [API 문서](#-api-문서)
- [프로젝트 구조](#-프로젝트-구조)
- [기술 스택](#-기술-스택)

## 🎯 프로젝트 개요

MatchMaker는 사용자의 프로필 정보, 위치, 선호도를 종합적으로 분석하여 가장 적합한 파트너를 추천하는 AI 기반 매칭 시스템입니다.

### ✨ 주요 특징

- **2단계 필터링 시스템**: Hard Filter + Soft Filter
- **AI 기반 텍스트 유사도**: SentenceTransformer를 활용한 태그 매칭
- **지리적 위치 기반**: 거리 계산을 통한 현실적인 매칭
- **다차원 점수 계산**: 6가지 요소를 종합한 매칭 점수

## 🚀 주요 기능

### 1. Hard Filtering (1차 필터링)

- **나이 조건**: 사용자가 설정한 최소/최대 나이 범위
- **거리 조건**: GPS 좌표 기반 최대 허용 거리
- **기본 조건**: 성별, 키 등 필수 조건

### 2. Soft Filtering (2차 필터링)

- **체형 매칭**: 체형 선호도 기반 점수 계산
- **학력 매칭**: 학력 수준 유사도 평가
- **종교 매칭**: 종교적 배경 고려
- **생활 패턴**: 흡연/음주 습관 매칭
- **취미/관심사**: AI 기반 태그 유사도 분석

### 3. 중복 방지 시스템

- **Mutual Exclusion**: 이미 매칭된 사용자 제외
- **이력 관리**: 매칭 기록 추적 및 관리

## 🏗 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Process Layer  │    │  Output Layer   │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • user_info     │───▶│ • preprocess.py │───▶│ • user_matching │
│ • user_cert     │    │ • soft_filter.py│    │ • db_util.py    │
│ • user_profile  │    │ • main.py       │    │                 │
│ • user_login    │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd MatchMakerProject

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

`data/` 폴더에 다음 CSV 파일들이 필요합니다:

- `user_info.csv`: 사용자 기본 정보 및 선호도
- `user_cert.csv`: 사용자 인증 정보 (성별, 생년월일)
- `user_profile.csv`: 사용자 프로필 (체형, 학력, 취미 등)
- `user_login_info.csv`: 위치 정보 (위도, 경도)
- `user_matching.csv`: 매칭 결과 저장

### 3. 실행

```bash
cd src
python main.py
```

## 📊 데이터 구조

### user_info

| 필드                   | 타입 | 설명               |
| ---------------------- | ---- | ------------------ |
| reg_no                 | int  | 사용자 고유번호    |
| distance               | int  | 최대 허용 거리(km) |
| min_age, max_age       | int  | 선호 나이 범위     |
| min_height, max_height | int  | 선호 키 범위       |

### user_profile

| 필드             | 타입 | 설명                         |
| ---------------- | ---- | ---------------------------- |
| user_no          | int  | 사용자 번호                  |
| body_type        | int  | 체형 (1:슬림~6:통통)         |
| academic_ability | int  | 학력 (1:고졸~5:대학원)       |
| religion         | int  | 종교 (1:무교~5:기타)         |
| smoking          | int  | 흡연 (1:흡연, 2:비흡연)      |
| drinking         | int  | 음주 (1:안마셔요~4:즐기는편) |
| me_tag           | str  | 해시태그 (#친절한#열정적인)  |

## 🧮 매칭 알고리즘

### 점수 계산 공식

```python
total_score = (
    body_type_score +      # 체형 유사도
    academic_score +       # 학력 유사도  
    religion_score +       # 종교 유사도
    smoking_score +        # 흡연 유사도
    drinking_score +       # 음주 유사도
    tag_similarity_score   # AI 태그 유사도
)
```

### 개별 점수 계산

1. **수치형 점수** (체형, 학력, 종교, 흡연, 음주)

   ```python
   score = 1.0 - (distance / max_distance)
   ```
2. **태그 유사도** (AI 기반)

   ```python
   # SentenceTransformer를 사용한 의미적 유사도
   similarity = model.similarity(user_embedding, candidate_embedding)
   score = similarity.mean()
   ```

### 필터링 기준

- **Hard Filter**: 나이, 거리 조건 불만족 시 완전 제외
- **Soft Filter**: 총점 0.5점 이상인 후보자만 선별

## 📁 프로젝트 구조

```
MatchMakerProject/
├── data/                    # 데이터 파일들
│   ├── user_info.csv
│   ├── user_cert.csv
│   ├── user_profile.csv
│   ├── user_login_info.csv
│   ├── user_matching.csv
│   ├── 01설명.txt          # 데이터 스키마 설명
│   └── 02쿼리.txt          # SQL 스키마
├── src/                     # 소스 코드
│   ├── main.py             # 메인 실행 파일
│   ├── preprocess.py       # 데이터 전처리 및 Hard Filter
│   ├── soft_filter.py      # Soft Filter 및 점수 계산
│   └── db_util.py          # 데이터베이스 유틸리티
├── requirements.txt         # 의존성 패키지
└── README.md               # 프로젝트 문서
```

## 🔧 기술 스택


### 파이썬 버전

3.13.5

### 핵심 라이브러리

- **pandas**: 데이터 처리 및 분석
- **sentence-transformers**: AI 기반 텍스트 유사도 계산
- **geopy**: 지리적 거리 계산
- **numpy**: 수치 계산

### AI/ML 모델

- **SentenceTransformer**: `jhgan/ko-sroberta-multitask` (한국어 특화)
- **코사인 유사도**: 텍스트 벡터 간 유사도 측정

### 데이터 처리

- **CSV 기반**: 가벼운 데이터 저장 및 처리
- **DataFrame**: 효율적인 테이블 데이터 조작📄 라이선스
