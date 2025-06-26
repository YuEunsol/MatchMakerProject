import pandas as pd
import os
from datetime import datetime, timedelta

def insert_dataframe_to_user_matching(df_candidates: pd.DataFrame, user_no: int, data_folder: str = "../data") -> int:
    """
    DataFrame 형태의 후보자 데이터를 user_matching.csv에 삽입합니다.
    
    Args:
        df_candidates (pd.DataFrame): 후보자 DataFrame (total_score 컬럼 포함)
        user_no (int): 매칭 요청 사용자 번호
        data_folder (str): 데이터 폴더 경로
    
    Returns:
        int: 삽입된 레코드 수
    """
    try:
        csv_path = os.path.join(data_folder, "user_matching.csv")
        
        # 기존 데이터 로드
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            # 새로운 reg_no 시작점 계산
            start_reg_no = existing_df['reg_no'].max() + 1
        else:
            # 파일이 없으면 새로 생성
            existing_df = pd.DataFrame()
            start_reg_no = 1
        
        # 현재 시간 설정
        current_time = datetime.now()
        reg_date = current_time.strftime("%Y-%m-%d %H:%M:%S")
        start_date = current_time.strftime("%Y-%m-%d %H:%M:%S")
        end_date = (current_time + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        view_end_date = (current_time + timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        
        # 새로운 매칭 레코드들 생성
        new_records = []
        
        for idx, candidate in df_candidates.iterrows():
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
            }
            new_records.append(new_record)
        
        # 새로운 레코드들을 DataFrame으로 변환
        new_df = pd.DataFrame(new_records)
        
        # 기존 데이터와 합치기
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        # CSV 파일에 저장
        combined_df.to_csv(csv_path, index=False)
        
        print(f"DataFrame 삽입 완료: User {user_no}에 대해 {len(new_records)}개 매칭 데이터 추가")
        return len(new_records)
        
    except Exception as e:
        print(f"DataFrame 삽입 중 오류 발생: {e}")
        return 0

