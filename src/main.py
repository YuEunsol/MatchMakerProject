import preprocess

if __name__ == '__main__':
    data_dir = "../data"
    df = preprocess.load_all_csvs(data_dir)
    matching_df = preprocess.load_matching_csv(data_dir)
    df = preprocess.add_age_column(df)
    female_df, male_df = preprocess.split_by_gender(df)

    for user in female_df.iterrows():
        preprocess.hard_filter(user, male_df)
        """
        이 부분에 소프트 필터링으로 점수 계산해서 matching_df에 작성하는 코드 추가해주시면 됩니다.
        """

    for user in male_df.iterrows():
        preprocess.hard_filter(user, female_df)
        """
        이 부분에 소프트 필터링으로 점수 계산해서 matching_df에 작성하는 코드 추가해주시면 됩니다.
        """