import pandas as pd
from preprocess import load_matching_csv
from sentence_transformers import SentenceTransformer

class SoftFilter:
    def __init__(self,user: pd.Series,model: SentenceTransformer, data_dir: str, mode: str):
        self.user = user
        self.model = model
        self.data_dir = data_dir
        self.mode = mode

    def mutualExclusionFilter(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """
            이미 매칭된 유저를 제외합니다.
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
        :return:
        """
        filter_candidates = []
        
        for _,candidate in candidates.iterrows():
            if mode == "mutual":
                score_ab = self._calculate_total_score(self.user, candidate)
                score_ba = self._calculate_total_score(candidate, self.user)
                total_score = (score_ab + score_ba) / 2
                print(
                    f"[mutual] {self.user.user_no} ↔ {candidate.user_no}: A→B={score_ab:.3f}, B→A={score_ba:.3f}, Total={total_score:.3f}")

            else:
                score_ab = self._calculate_total_score(self.user, candidate)
                score_ba = None
                total_score = score_ab

            if total_score > 3:
                candidate_dict = candidate.to_dict()
                candidate_dict["score_a_to_b"] = round(score_ab, 3) if score_ab is not None else None
                candidate_dict["score_b_to_a"] = round(score_ba, 3) if score_ba is not None else None
                candidate_dict["total_score"] = round(total_score, 3) if total_score is not None else None
                filter_candidates.append(candidate_dict)

        if not filter_candidates:
            return pd.DataFrame(columns=list(candidates.columns) + ["score_a_to_b", "score_b_to_a", "total_score"])

        return pd.DataFrame(filter_candidates)[
            list(candidates.columns) + ["score_a_to_b", "score_b_to_a", "total_score"]].reset_index(drop=True)

    def _calculateBodyTypeMatchScore(self, user: pd.Series, candidate: pd.Series) -> float:
        """
            체형 점수 계산
        """
        max_distance = 6
        distance = abs(user.body_type - candidate.body_type)
        score = 1.0 - (distance / max_distance)
        return score

    def _calculateAcademicMatchScore(self, user: pd.Series, candidate: pd.Series) -> float:
        """
            학력 점수 계산
        """
        max_distance = 5
        distance = abs(user.academic_ability - candidate.academic_ability)
        score = 1.0 - (distance / max_distance)
        return score
    
    def _calcluateReligionMatchScore(self, user: pd.Series, candidate: pd.Series) -> float:
        """
            종교 점수 계산
        """
        max_distance = 5
        distance = abs(user.religion - candidate.religion)
        score = 1.0 - (distance / max_distance)
        return score
    
    def _calculateSmokingMatchScore(self, user: pd.Series, candidate: pd.Series) -> float:
        """
            흡연 점수 계산
        """
        if user.smoking == candidate.smoking:
            return 1.0
        else:
            return 0.0
        

    def _calculateDrinkingMatchScore(self, user: pd.Series, candidate: pd.Series) -> float:
        """
            음주 점수 계산
        """
        max_distance = 4
        distance = abs(user.drinking - candidate.drinking)
        score = 1.0 - (distance / max_distance)
        return score
    
    def _calculateTagSimilarityScore(self, user: pd.Series, candidate: pd.Series) -> float:
        """
            태그 유사도 점수 계산
        """
        
        user_tags = user.me_tag.split("#")[1:]  # 첫 번째 빈 문자열 제거
        candidate_tags = candidate.me_tag.split("#")[1:]

        # 태그가 없는 경우 처리
        if not user_tags or not candidate_tags:
            return 0.0

        user_embedding = self.model.encode(user_tags)
        candidate_embedding = self.model.encode(candidate_tags)

        # 유사도 매트릭스 계산
        similarity_matrix = self.model.similarity(user_embedding, candidate_embedding)
        
        # 평균값 방식: 모든 유사도의 평균
        avg_similarity = float(similarity_matrix.mean())
        
        return avg_similarity

    def _calculate_total_score(self, user: pd.Series, candidate: pd.Series) -> float:
        body_type_score = self._calculateBodyTypeMatchScore(user, candidate)
        academic_score = self._calculateAcademicMatchScore(user, candidate)
        religion_score = self._calcluateReligionMatchScore(user, candidate)
        smoking_score = self._calculateSmokingMatchScore(user, candidate)
        drinking_score = self._calculateDrinkingMatchScore(user, candidate)
        tag_score = self._calculateTagSimilarityScore(user, candidate)

        return body_type_score + academic_score + religion_score + smoking_score + drinking_score + tag_score






    

