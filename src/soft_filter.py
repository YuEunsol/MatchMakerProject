import pandas as pd
from preprocess import load_matching_csv
from sentence_transformers import SentenceTransformer

class SoftFilter:
    def __init__(self,user: pd.Series,model: SentenceTransformer):
        self.user = user
        self.model = model

    def mutualExclusionFilter(self,candidates: pd.DataFrame) -> pd.DataFrame:
        """
            이미 매칭된 유저를 제외합니다.
        """
        user_no = self.user.user_no
        not_recommended_candidates =[]
        
        for _,candidate in candidates.iterrows():
            flag = False
            for _,match_data in load_matching_csv("../data").iterrows():
                if match_data.user_no == user_no and match_data.rec_user_no == candidate.user_no:
                    flag = True
                    break
            if flag:
                continue
            else:
                not_recommended_candidates.append(candidate)
        
        return pd.DataFrame(not_recommended_candidates)
    
    def profileMatchScoreFilter(self,candidates: pd.DataFrame) -> pd.DataFrame:
        """
            프로필 점수 계산
        """
        filter_candidates = []
        
        for _,candidate in candidates.iterrows():
            body_type_score = self._calculateBodyTypeMatchScore(candidate)
            academic_score = self._calculateAcademicMatchScore(candidate)
            religion_score = self._calcluateReligionMatchScore(candidate)
            smoking_score = self._calculateSmokingMatchScore(candidate)
            drinking_score = self._calculateDrinkingMatchScore(candidate)
            tag_similarity_score = self._calculateTagSimilarityScore(candidate)
            total_score = body_type_score + academic_score + religion_score + smoking_score + drinking_score + tag_similarity_score
            if total_score > 0.5:
                filter_candidates.append(candidate)

        return pd.DataFrame(filter_candidates)

    def _calculateBodyTypeMatchScore(self,candidate: pd.Series) -> float:
        """
            체형 점수 계산
        """
        max_distance = 6
        distance = abs(self.user.body_type - candidate.body_type)
        score = 1.0 - (distance / max_distance)
        return score
    
    def _calculateAcademicMatchScore(self,candidate: pd.Series) -> float:
        """
            학력 점수 계산
        """
        max_distance = 5
        distance = abs(self.user.academic_ability - candidate.academic_ability)
        score = 1.0 - (distance / max_distance)
        return score
    
    def _calcluateReligionMatchScore(self,candidate: pd.Series) -> float:
        """
            종교 점수 계산
        """
        max_distance = 5
        distance = abs(self.user.religion - candidate.religion)
        score = 1.0 - (distance / max_distance)
        return score
    
    def _calculateSmokingMatchScore(self,candidate: pd.Series) -> float:
        """
            흡연 점수 계산
        """
        max_distance = 5
        distance = abs(self.user.smoking - candidate.smoking)
        score = 1.0 - (distance / max_distance)
        return score

    def _calculateDrinkingMatchScore(self,candidate: pd.Series) -> float:
        """
            음주 점수 계산
        """
        max_distance = 4
        distance = abs(self.user.drinking - candidate.drinking)
        score = 1.0 - (distance / max_distance)
        return score
    
    def _calculateTagSimilarityScore(self,candidate: pd.Series) -> float:
        """
            태그 유사도 점수 계산
        """
        
        user_tags = self.user.me_tag.split("#")[1:]  # 첫 번째 빈 문자열 제거
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

    





    

