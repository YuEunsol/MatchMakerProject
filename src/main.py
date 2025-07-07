import os
import time
import preprocess
import pandas as pd
from soft_filter import SoftFilter
from db_util import insert_dataframe_to_user_matching
from sentence_transformers import SentenceTransformer

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, "../data"))

    df = preprocess.load_all_csvs(data_dir)
    matching_df = preprocess.load_matching_csv(data_dir)
    df = preprocess.add_age_column(df)
    female_df, male_df = preprocess.split_by_gender(df)

    model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    for score_mode in ["single", "mutual"]:
        print(f"\n===== [{score_mode.upper()}] 매칭 시작 =====")
        start_time = time.time()

        for _, user in female_df.iterrows():
            filtered_df = preprocess.hard_filter(user, male_df)
            soft_filter = SoftFilter(user, model, data_dir, mode=score_mode)
            filtered_df = soft_filter.mutualExclusionFilter(filtered_df)
            filtered_df = soft_filter.profileMatchScoreFilter(filtered_df, mode=score_mode)
            print(f"[{score_mode}] {user.user_no} After filter: {len(filtered_df)}")
            insert_dataframe_to_user_matching(filtered_df, user.user_no, data_dir, mode=score_mode)

        for _, user in male_df.iterrows():
            filtered_df = preprocess.hard_filter(user, female_df)
            soft_filter = SoftFilter(user, model, data_dir, mode=score_mode)
            filtered_df = soft_filter.mutualExclusionFilter(filtered_df)
            filtered_df = soft_filter.profileMatchScoreFilter(filtered_df, mode=score_mode)
            insert_dataframe_to_user_matching(filtered_df, user.user_no, data_dir, mode=score_mode)

        elapsed = time.time() - start_time
        print(f"[{score_mode.upper()}] 매칭 완료: {elapsed:.2f}초 소요")

