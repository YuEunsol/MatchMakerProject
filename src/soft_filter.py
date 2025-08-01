import pandas as pd
from preprocess import load_matching_csv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SoftFilter:
    def __init__(self,user: pd.Series, data_dir: str, mode: str):
        self.user = user
        self.data_dir = data_dir
        self.mode = mode
         # 태그 카테고리 매핑 정의
        self.tag_categories = {
            # 성격 카테고리
            '지성형': ['지적인', '창의적인', '진지한', '과묵한', '차분한', '조용한'],
            '감성형': ['다정다감', '잘웃는', '섬세한', '애교있는', '친절한'],
            '활동형': ['열정적인', '활발한', '시원한', '남자다운', '털털한', '발랄한'],
            '긍정형': ['긍정적인', '낙천적인', '자신감있는', '재미있는', '센스있는'],
            '성숙형': ['어른스러운', '한결같은', '도도한'],
            '개성형': ['귀여운', '장난기많은', '호기심많은'],
            
            # 관심사 카테고리
            '문화생활': ['독서', '영화', '음악', '전시회', '카페'],
            '활동취미': ['여행', '스포츠/운동', '피트니스', '드라이브', '노래'],
            '일상취미': ['맛집', '쇼핑', '사진', '휴식', '음주'],
            '자기계발': ['재테크', '시사', '외국어', '자기계발'],
            '기타취미': ['반려동물', '게임']
        }
        self.personality_categories = ['지성형', '감성형', '활동형', '긍정형', '성숙형', '개성형']
        self.interest_categories = ['문화생활', '활동취미', '일상취미', '자기계발', '기타취미']

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
            태그 사전 분류 기반 유사도 점수 계산
            태그를 사전 정의된 성격 카테고리로 매핑 후 유사도 계산
            args:
                user: 유저 데이터 열
                candidate: 후보군 데이터프레임
            return:
                태그 유사도 점수
        """
        
        # 역방향 매핑 (태그 -> 카테고리)
        tag_to_category = {}
        for category, tags in self.tag_categories.items():
            for tag in tags:
                tag_to_category[tag] = category

        user_tags = [tag.strip() for tag in user.me_tag.split("#")[1:] if tag.strip()]
        candidate_tags = [tag.strip() for tag in candidate.me_tag.split("#")[1:] if tag.strip()]

        if not user_tags or not candidate_tags:
            return 0.0

        # 각 사용자의 카테고리 분포 계산
        user_categories = {}
        candidate_categories = {}
        
        # 사용자 태그의 카테고리 분포
        for tag in user_tags:
            if tag in tag_to_category:
                category = tag_to_category[tag]
                user_categories[category] = user_categories.get(category, 0) + 1
        
        # 후보자 태그의 카테고리 분포
        for tag in candidate_tags:
            if tag in tag_to_category:
                category = tag_to_category[tag]
                candidate_categories[category] = candidate_categories.get(category, 0) + 1
        
        # 카테고리가 없는 경우 처리
        if not user_categories or not candidate_categories:
            return 0.0
        
        # 카테고리 매칭 점수 계산
        all_categories = set(user_categories.keys()) | set(candidate_categories.keys())
        
        # 카테고리별 벡터 생성 (정규화)
        user_vector = []
        candidate_vector = []
        
        for category in all_categories:
            user_count = user_categories.get(category, 0)
            candidate_count = candidate_categories.get(category, 0)
            
            # 각 카테고리의 비율로 정규화
            user_ratio = user_count / len(user_tags)
            candidate_ratio = candidate_count / len(candidate_tags)
            
            # 성격 카테고리에 더 높은 가중치 부여 (성격: 0.7, 관심사: 0.3)
            if category in self.personality_categories:
                user_ratio *= 0.7
                candidate_ratio *= 0.7
            else:
                user_ratio *= 0.3
                candidate_ratio *= 0.3
            
            user_vector.append(user_ratio)
            candidate_vector.append(candidate_ratio)
        
        # 코사인 유사도 계산
        user_vector = np.array(user_vector).reshape(1, -1)
        candidate_vector = np.array(candidate_vector).reshape(1, -1)
        
        similarity = cosine_similarity(user_vector, candidate_vector)[0][0]
        
        # NaN이나 Inf 값 처리
        if np.isnan(similarity) or np.isinf(similarity):
            return 0.0
            
        return float(similarity)

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


# if __name__ == "__main__":
#     # 테스트 데이터 생성
#     user_data = pd.Series({'me_tag': '#지적인#창의적인#독서#영화'})
#     candidate_data = pd.Series({'me_tag': '#차분한#친절한#카페#음악'})


#     # SoftFilter 인스턴스 생성
#     soft_filter = SoftFilter(user_data, "./data", "single")

#     # 태그 유사도 점수 계산
#     similarity_score = soft_filter._calculateTagSimilarityScore(user_data, candidate_data)
#     print(f"태그 유사도 점수: {similarity_score}")






    

