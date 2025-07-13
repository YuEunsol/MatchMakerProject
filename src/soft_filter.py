import pandas as pd
from preprocess import load_matching_csv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SoftFilter:
    def __init__(self,user: pd.Series,model: SentenceTransformer, data_dir: str, mode: str):
        self.user = user
        self.model = model
        self.data_dir = data_dir
        self.mode = mode

    def mutualExclusionFilter(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """
            이미 매칭된 유저를 제외합니다.
            args:
                candidates: 후보군 데이터프레임
            return:
                not_recommended_candidates: 이미 매칭된 유저를 제외한 후보군 데이터프레임
        """
        user_no = self.user.user_no
        not_recommended_candidates =[]
        
        for _,candidate in candidates.iterrows():
            flag = False
            for _,match_data in load_matching_csv(self.data_dir).iterrows():
                if match_data.user_no == user_no and match_data.rec_user_no == candidate.user_no and match_data.score_mode == self.mode:
                    flag = True
                    break
            if flag:
                continue
            else:
                not_recommended_candidates.append(candidate)
        
        return pd.DataFrame(not_recommended_candidates)
    
    def profileMatchScoreFilter(self,candidates: pd.DataFrame, mode: str) -> pd.DataFrame:
        """
        프로필 기반 점수 필터링
        :param candidates: 계산 대상
        :param mode: 스코어링 모드
            - mode="single" : 단일 스코어링. self.user → candidate 기준 단방향
            - mode="mutual" : 쌍방 스코어링.
        :return: 스코어링 후 필터링된 후보군 데이터프레임
        """
        filter_candidates = []

        for _, candidate in candidates.iterrows():
            if mode == "mutual":
                # user_profile column 데이터를 기반으로 스코어링 (자세한 로직은 수도 코드의 soft_filter 참조)
                score_ab = self._calculate_total_score(self.user, candidate)
                score_ba = self._calculate_total_score(candidate, self.user)

                # 스코어가 3 이상인 경우만 통과
                if score_ab > 3 and score_ba > 3:
                    total_score = (score_ab + score_ba) / 2
                else:
                    continue

                print(
                    f"[mutual] {self.user.user_no} ↔ {candidate.user_no}: A→B={score_ab:.3f}, B→A={score_ba:.3f}, Total={total_score:.3f}")

            else:
                # user_profile column 데이터를 기반으로 스코어링 (자세한 로직은 수도 코드의 soft_filter 참조)
                score_ab = self._calculate_total_score(self.user, candidate)
                score_ba = None
                # 스코어가 3 이상인 경우만 통과
                if score_ab > 3:
                    total_score = score_ab
                else:
                    continue

            candidate_dict = candidate.to_dict()
            candidate_dict["score_a_to_b"] = round(score_ab, 3) if score_ab is not None else None
            candidate_dict["score_b_to_a"] = round(score_ba, 3) if score_ba is not None else None
            candidate_dict["total_score"] = round(total_score, 3)
            filter_candidates.append(candidate_dict)

        if not filter_candidates:
            return pd.DataFrame(columns=list(candidates.columns) + ["score_a_to_b", "score_b_to_a", "total_score"])

        return pd.DataFrame(filter_candidates)[
            list(candidates.columns) + ["score_a_to_b", "score_b_to_a", "total_score"]].reset_index(drop=True)

    def _calculateBodyTypeMatchScore(self, user: pd.Series, candidate: pd.Series) -> float:
        """
            체형 점수 계산
            args:
                user: 유저 데이터 열
                candidate: 후보군 데이터프레임
            return:
                체형 점수
        """
        max_distance = 6
        distance = abs(user.body_type - candidate.body_type)
        score = 1.0 - (distance / max_distance)
        return score

    def _calculateAcademicMatchScore(self, user: pd.Series, candidate: pd.Series) -> float:
        """
            학력 점수 계산
            args:
                user: 유저 데이터 열
                candidate: 후보군 데이터프레임
            return:
                학력 점수
        """
        max_distance = 5
        distance = abs(user.academic_ability - candidate.academic_ability)
        score = 1.0 - (distance / max_distance)
        return score
    
    def _calcluateReligionMatchScore(self, user: pd.Series, candidate: pd.Series) -> float:
        """
            종교 점수 계산
            args:
                user: 유저 데이터 열
                candidate: 후보군 데이터프레임
            return:
                종교 점수
        """
        max_distance = 5
        distance = abs(user.religion - candidate.religion)
        score = 1.0 - (distance / max_distance)
        return score
    
    def _calculateSmokingMatchScore(self, user: pd.Series, candidate: pd.Series) -> float:
        """
            흡연 점수 계산
            args:
                user: 유저 데이터 열
                candidate: 후보군 데이터프레임
            return:
                흡연 점수
        """
        if user.smoking == candidate.smoking:
            return 1.0
        else:
            return 0.0
        

    def _calculateDrinkingMatchScore(self, user: pd.Series, candidate: pd.Series) -> float:
        """
            음주 점수 계산
            args:
                user: 유저 데이터 열
                candidate: 후보군 데이터프레임
            return:
                음주 점수
        """
        max_distance = 4
        distance = abs(user.drinking - candidate.drinking)
        score = 1.0 - (distance / max_distance)
        return score
    
    def _calculateTagSimilarityScore(self, user: pd.Series, candidate: pd.Series) -> float:
        """
            태그 유사도 점수 계산
            args:
                user: 유저 데이터 열
                candidate: 후보군 데이터프레임
            return:
                태그 유사도 점수
        """

        user_tags = user.me_tag.split("#")[1:]
        candidate_tags = candidate.me_tag.split("#")[1:]

        if not user_tags or not candidate_tags:
            return 0.0

        user_embedding = self.model.encode(user_tags)
        candidate_embedding = self.model.encode(candidate_tags)

        sim_matrix = cosine_similarity(user_embedding, candidate_embedding)
        # user 태그 기준으로 가장 유사한 candidate 태그 하나씩만 고름
        max_similarities = sim_matrix.max(axis=1)
        avg_max_similarity = float(np.mean(max_similarities))

        return avg_max_similarity

    def _calculate_total_score(self, user: pd.Series, candidate: pd.Series) -> float:
        """
            총 점수 계산
            args:
                user: 유저 데이터 열
                candidate: 후보군 데이터프레임
            return:
                총 점수
        """
        body_type_score = self._calculateBodyTypeMatchScore(user, candidate)
        academic_score = self._calculateAcademicMatchScore(user, candidate)
        religion_score = self._calcluateReligionMatchScore(user, candidate)
        smoking_score = self._calculateSmokingMatchScore(user, candidate)
        drinking_score = self._calculateDrinkingMatchScore(user, candidate)
        tag_score = self._calculateTagSimilarityScore(user, candidate)

        return body_type_score + academic_score + religion_score + smoking_score + drinking_score + tag_score






    

