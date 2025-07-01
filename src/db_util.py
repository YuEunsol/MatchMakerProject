import pandas as pd
import os
from datetime import datetime, timedelta

def insert_dataframe_to_user_matching(df_candidates: pd.DataFrame, user_no: int, data_folder: str, mode: str) -> int:
    """
    DataFrame 형태의 후보자 데이터를 user_matching.csv에 삽입합니다.
    후보자 데이터를 user_matching.csv에 통합 저장하고, score_mode로 단방향/쌍방향 구분
    
    Args:
        df_candidates (pd.DataFrame): 후보자 DataFrame (total_score 컬럼 포함)
        user_no (int): 매칭 요청 사용자 번호
        data_folder (str): 데이터 폴더 경로
    
    Returns:
        int: 삽입된 레코드 수
    """
    try:
        file_path = os.path.join(data_folder, "user_matching.csv")
        csv_path = os.path.abspath(file_path)

        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            existing_df = pd.read_csv(csv_path)

            # score_mode 컬럼이 없으면 기본값으로 채워줌
            if 'score_mode' not in existing_df.columns:
                existing_df['score_mode'] = 'unknown'

            start_reg_no = existing_df['reg_no'].max() + 1

            # 중복 체크: (user_no, rec_user_no, score_mode) 조합
            existing_pairs = set(zip(
                existing_df["user_no"],
                existing_df["rec_user_no"],
                existing_df["score_mode"]
            ))
        else:
            existing_df = pd.DataFrame()
            start_reg_no = 1
            # 중복 체크용 (같은 user_no + rec_user_no + score_mode)
            existing_pairs = set()
        
        # 현재 시간 설정
        current_time = datetime.now()
        reg_date = current_time.strftime("%Y-%m-%d %H:%M:%S")
        start_date = current_time.strftime("%Y-%m-%d %H:%M:%S")
        end_date = (current_time + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        view_end_date = (current_time + timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        
        # 새로운 매칭 레코드들 생성
        new_records = []

        for idx, candidate in df_candidates.iterrows():
            key = (user_no, candidate['user_no'], mode)
            if key in existing_pairs:
                continue

            new_record = {
                'reg_no': start_reg_no + len(new_records),
                'user_no': user_no,
                'rec_user_no': candidate['user_no'],
                'reg_date': reg_date,
                'start_date': start_date,
                'end_date': end_date,
                'status': 'P',  # P: Pending
                'view': 'N',    # N: Not viewed
                'view_end_date': view_end_date,
                'del_yn': 'N',  # N: Not deleted
                'del_date': None,
                'score_mode': mode,
                'score_a_to_b': candidate["score_a_to_b"] if "score_a_to_b" in candidate else None,
                'score_b_to_a': candidate["score_b_to_a"] if "score_b_to_a" in candidate else None,
                'total_score': candidate["total_score"] if "total_score" in candidate else None,

            }
            new_records.append(new_record)

        if not new_records:
            print(f"DataFrame 삽입 완료: User {user_no}에 대해 0개 매칭 데이터 추가 (중복 제외됨)")
            return 0

        new_df = pd.DataFrame(new_records)
        final_df = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
        final_df.to_csv(csv_path, index=False)
        
        print(f"DataFrame 삽입 완료: User {user_no}에 대해 {len(new_records)}개 매칭 데이터 추가")
        return len(new_records)
        
    except Exception as e:
        print(f"DataFrame 삽입 중 오류 발생: {e}")
        return 0

