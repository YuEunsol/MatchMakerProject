import os
import pandas as pd
from datetime import datetime
from geopy.distance import geodesic

def print_df(df):
    """
    테이블 확인용 출력
    :param df: 테이블
    :return: None
    """
    # 출력 설정
    pd.set_option('display.max_columns', None)  # 모든 컬럼 출력
    pd.set_option('display.width', 1000)  # 줄바꿈 안 되게
    pd.set_option('display.max_colwidth', None)  # 셀 내용 자르지 않게

    # print(df)


def load_all_csvs(folder_path):
    """
    user_info, user_cert, user_profile, user_login_info 테이블을 하나로 병합
    :param folder_path: csv 파일이 있는 폴더 경로
    :return df: 병합한 테이블
    """
    user_info = pd.read_csv(os.path.join(folder_path, "user_info.csv"))
    user_cert = pd.read_csv(os.path.join(folder_path, "user_cert.csv"))
    user_profile = pd.read_csv(os.path.join(folder_path, "user_profile.csv"))
    user_login_info = pd.read_csv(os.path.join(folder_path, "user_login_info.csv"))

    df = user_cert.merge(user_info, on='reg_no') \
              .merge(user_profile, on='user_no') \
              .merge(user_login_info, on='user_no')
    
    # user_profile의 중복데이터로 인해 merge 시 중복데이터가 발생.
    # user_no 기준으로 중복 데이터 제거 (첫 번째 발견된 데이터만 유지)
    df = df.drop_duplicates(subset=['user_no'], keep='first')

    # 명시적으로 컬럼명 변경
    df = df.rename(columns={
        'reg_no_x': 'user_info_reg_no',
        'reg_no_y': 'user_profile_reg_no'
    })

    return df

def load_matching_csv(folder_path):
    """
    user_matching 테이블 가져오기
    :param folder_path: csv 파일이 있는 폴더 경로
    :return matching_df: user_matching 테이블
    """
    return pd.read_csv(os.path.join(folder_path, "user_matching.csv"))

def add_age_column(df, birth_col="birth_date", age_col="age"):
    """
    생년월일 기반 한국 나이 계산하여 테이블에 추가
    :param df: 병합된 테이블
    :param birth_col: 생년월일 컬럼
    :param age_col: 나이 컬럼
    :return: 나이 추가된 테이블
    """
    current_year = datetime.now().year

    def calculate_age(birth):
        birth = str(int(birth))  # 혹시 float 형태일 경우 대비
        year = int(birth[:4])
        return current_year - year

    df[age_col] = df[birth_col].apply(calculate_age)

    # print_df(df)

    return df

def split_by_gender(df):
    """
    남, 여 테이블 분리
    :param df: 병합된 테이블
    :return female_df, male_df: 남, 여 테이블
    """
    female_df = df[df["gender"] == 0]
    male_df = df[df["gender"] == 1]

    # print_df(female_df)
    # print_df(male_df)

    return female_df, male_df

def calc_distance(coord1, coord2):
    """
    두 좌표간 거리(km) 계산
    :param coord1: 좌표1
    :param coord2: 좌표2
    :return: 거리값(km)
    """
    return geodesic(coord1, coord2).km

def hard_filter(user, candidates):
    """
    나이, 거리 기반 하드 필터링
    :param user: 검색 대상 유저
    :param candidates: 비교 대상 유저
    :return filtered_df: 하드 필터링 결과 테이블
    """
    filtered = []

    for _, candidate in candidates.iterrows():

        # 나이 조건
        if not (user['min_age'] <= candidate['age'] <= user['max_age']):
            continue

        # 예외 방지: 좌표 누락된 경우
        if pd.isna(user['latitude']) or pd.isna(candidate['latitude']):
            continue

        # 거리 계산
        my_coords = (user['latitude'], user['longitude'])
        cand_coords = (candidate['latitude'], candidate['longitude'])
        dist = calc_distance(my_coords, cand_coords)
        if dist > user['distance']:
            continue

        filtered.append(candidate)

    filtered_df = pd.DataFrame(filtered)
    # print_df(filtered_df)

    return filtered_df

# df = load_all_csvs("../data")
# df = add_age_column(df)
# female_df, male_df = split_by_gender(df)
# hard_filter(female_df.iloc[0], male_df)