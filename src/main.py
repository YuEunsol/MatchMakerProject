import os
import time
import preprocess
import pandas as pd
from soft_filter import SoftFilter
from db_util import insert_dataframe_to_user_matching

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, "../data"))

    # 데이터 로드
    df = preprocess.load_all_csvs(data_dir)
    matching_df = preprocess.load_matching_csv(data_dir)
    # 나이 추가
    df = preprocess.add_age_column(df)
    # 남/여를 기준으로 데이터 프레임 분리
    female_df, male_df = preprocess.split_by_gender(df)

    # 매칭 시작
    for score_mode in ["single", "mutual"]:
        print(f"\n===== [{score_mode.upper()}] 매칭 시작 =====")
        start_time = time.time()

        for _, user in female_df.iterrows():
            # user_info 이상형 기준으로 필터링 (자세한 로직은 수도 코드의 hard_filter 참조)
            filtered_df = preprocess.hard_filter(user, male_df)
            # user_profile 기준으로 스코어링하여 필터링 클래스로 부터 인스턴스 생성 (자세한 로직은 수도 코드의 soft_filter 참조)
            soft_filter = SoftFilter(user,data_dir, mode=score_mode)
            # user_matching 테이블에 존재하는 이미 매칭된 유저를 후보군에서 제외합니다.
            filtered_df = soft_filter.mutualExclusionFilter(filtered_df)
            # 프로필 스코어링 후 필터링 (자세한 로직은 수도 코드의 soft_filter 참조)
            filtered_df = soft_filter.profileMatchScoreFilter(filtered_df, mode=score_mode)
            print(f"[{score_mode}] {user.user_no} After filter: {len(filtered_df)}")
            # user_matching 테이블에 추가
            insert_dataframe_to_user_matching(filtered_df, user.user_no, data_dir, mode=score_mode)

        for _, user in male_df.iterrows():
            # user_info 이상형 기준으로 필터링 (자세한 로직은 수도 코드의 hard_filter 참조)
            filtered_df = preprocess.hard_filter(user, female_df)
            # user_profile 기준으로 스코어링하여 필터링 클래스로 부터 인스턴스 생성 (자세한 로직은 수도 코드의 soft_filter 참조)
            soft_filter = SoftFilter(user,data_dir, mode=score_mode)
            # user_matching 테이블에 존재하는 이미 매칭된 유저를 후보군에서 제외합니다.
            filtered_df = soft_filter.mutualExclusionFilter(filtered_df)
            # 프로필 스코어링 후 필터링 (자세한 로직은 수도 코드의 soft_filter 참조)
            filtered_df = soft_filter.profileMatchScoreFilter(filtered_df, mode=score_mode)
            # user_matching 테이블에 추가
            insert_dataframe_to_user_matching(filtered_df, user.user_no, data_dir, mode=score_mode)

        elapsed = time.time() - start_time
        print(f"[{score_mode.upper()}] 매칭 완료: {elapsed:.2f}초 소요")

