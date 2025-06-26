import preprocess
from soft_filter import SoftFilter
from db_util import insert_dataframe_to_user_matching
from sentence_transformers import SentenceTransformer

if __name__ == '__main__':
    data_dir = "../data"
    df = preprocess.load_all_csvs(data_dir)
    matching_df = preprocess.load_matching_csv(data_dir)
    df = preprocess.add_age_column(df)
    female_df, male_df = preprocess.split_by_gender(df)

    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    
    for _,user in female_df.iterrows():
        filtered_df = preprocess.hard_filter(user, male_df)
        """
        이 부분에 소프트 필터링으로 점수 계산해서 matching_df에 작성하는 코드 추가해주시면 됩니다.
        """
        soft_filter = SoftFilter(user,model)
        filtered_df = soft_filter.mutualExclusionFilter(filtered_df)
        filtered_df = soft_filter.profileMatchScoreFilter(filtered_df)
        insert_dataframe_to_user_matching(filtered_df, user.user_no)
        

    for _,user in male_df.iterrows():
        filtered_df = preprocess.hard_filter(user, female_df)
        """
        이 부분에 소프트 필터링으로 점수 계산해서 matching_df에 작성하는 코드 추가해주시면 됩니다.
        """
        soft_filter = SoftFilter(user,model)
        filtered_df = soft_filter.mutualExclusionFilter(filtered_df)
        filtered_df = soft_filter.profileMatchScoreFilter(filtered_df)
        insert_dataframe_to_user_matching(filtered_df, user.user_no)