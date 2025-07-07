import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# user_matching.csv 로딩
df = pd.read_csv("../data/user_matching.csv")

# single / mutual 분리
df_single = df[df["score_mode"] == "single"]
df_mutual = df[df["score_mode"] == "mutual"]

# 단방향 쌍들 set 생성
single_pairs = set(zip(df_single["user_no"], df_single["rec_user_no"]))
single_pairs_reverse = set((b, a) for (a, b) in single_pairs)

# mutual에서 (a, b) 혹은 (b, a) 둘 다 단일에 있는 경우만 남김
filtered_mutual_rows = []

for _, row in df_mutual.iterrows():
    a, b = row["user_no"], row["rec_user_no"]
    if (a, b) in single_pairs and (b, a) in single_pairs:
        filtered_mutual_rows.append(row)

# 결과 저장
df_filtered_mutual = pd.DataFrame(filtered_mutual_rows)
print(f"원래 mutual 쌍 수: {len(df_mutual)}")
print(f"짝 맞는 mutual 쌍만 남긴 수: {len(df_filtered_mutual)}")

# 최종적으로 single과 필터링된 mutual 합치기
df_final = pd.concat([df_single, df_filtered_mutual], ignore_index=True)

# 덮어쓰기 저장
df_final.to_csv("../data/user_matching.csv", index=False)


# 매칭 결과 csv 불러오기
df = pd.read_csv("../data/user_matching.csv")  # 결과 파일 경로

# single / mutual 구분
df_single = df[df["score_mode"] == "single"]
df_mutual = df[df["score_mode"] == "mutual"]

# 단방향 쌍 set
single_pairs = set(zip(df_single["user_no"], df_single["rec_user_no"]))
# mutual은 한 방향만 저장되어 있음
mutual_pairs = set(zip(df_mutual["user_no"], df_mutual["rec_user_no"]))
mutual_pairs_reversed = set((b, a) for a, b in mutual_pairs)

# mutual 쌍이 단일 결과에 들어있는지 체크
matched_pairs = single_pairs & mutual_pairs
reversed_matches = single_pairs & mutual_pairs_reversed
total_matched = matched_pairs | reversed_matches

print("==== 정확도 비교 ====")
print(f"단일 추천 쌍 총 수: {len(single_pairs)}")
print(f"쌍방향 추천 쌍 총 수: {len(mutual_pairs)}")
print(f"단일 추천 중 mutual에서 같은 방향으로 겹치는 쌍: {len(matched_pairs)}")
print(f"단일 추천 중 mutual에서 반대 방향으로 겹치는 쌍: {len(reversed_matches)}")
print(f"총 mutual과 짝이 맞는 단일 쌍 수: {len(total_matched)}")

print("\n==== 추천 건수 비교 ====")
print(f"Single 추천 수: {len(df_single)}")
print(f"Mutual 추천 수: {len(df_mutual)}")

print("\n==== 유저 수 비교 ====")
print(f"Single 추천한 유저 수: {df_single['user_no'].nunique()}")
print(f"Mutual 추천한 유저 수: {df_mutual['user_no'].nunique()}")

# 유저별 추천 수 통계
def get_stats(df, label):
    grouped = df.groupby("user_no").size()
    print(f"\n{label} 유저당 추천 수 통계")
    print(f"평균: {grouped.mean():.2f}")
    print(f"최대: {grouped.max()}")
    print(f"최소: {grouped.min()}")
    print(f"중간값: {grouped.median()}")
    return grouped

single_stats = get_stats(df_single, "Single")
mutual_stats = get_stats(df_mutual, "Mutual")

# 히스토그램
plt.hist(single_stats, bins=range(1, single_stats.max()+2), alpha=0.4, label='Single', color='skyblue', edgecolor='black')
plt.hist(mutual_stats, bins=range(1, mutual_stats.max()+2), alpha=0.4, label='Mutual', color='coral', edgecolor='black')
plt.legend()
plt.xlabel("유저당 추천 수")
plt.ylabel("유저 수")
plt.title("Single vs Mutual 추천 수 분포")
plt.grid(True)
plt.tight_layout()
plt.show()

# 추천된 쌍의 겹침 분석
intersection = single_pairs & mutual_pairs
print("\n==== 추천 쌍 겹침 분석 ====")
print(f"Single과 Mutual 모두에서 추천된 쌍 수: {len(intersection)}")
print(f"Single 대비 겹치는 비율: {len(intersection)/len(single_pairs)*100:.2f}%")
print(f"Mutual 대비 겹치는 비율: {len(intersection)/len(mutual_pairs)*100:.2f}%")

# 실행 시간 기반 확장 예상 (수동 입력)
single_time = 32.62  # 실제 실행 시간(초)
mutual_time = 37.31

def estimate_runtime(current_n_users: int, current_time_sec: float, target_n_users: int) -> float:
    scale_ratio = (target_n_users / current_n_users) ** 2
    return current_time_sec * scale_ratio

print("\n==== 예상 실행 시간 계산 ====")
for mode_name, df_mode, runtime in [("Single", df_single, single_time), ("Mutual", df_mutual, mutual_time)]:
    current_users = df_mode["user_no"].nunique()
    print(f"\n[{mode_name}] 현재 유저 수: {current_users}명, 실행 시간: {runtime:.2f}초")
    for target in [10_000, 50_000]:
        est_sec = estimate_runtime(current_users, runtime, target)
        print(f"{target:,}명일 경우 예상 실행 시간: {est_sec/60:.1f}분 ({est_sec:.1f}초)")
