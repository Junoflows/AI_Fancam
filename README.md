# 깊이 추정을 통한 AI 직캠 인물 검출 성능 향상
서울시립대와 KBS 미디어 기술 연구소와 함께 진행한 프로젝트로, 해당 주제로 작성한 논문입니다.
+ KOBA 2024 KBS 공개 AI 세미나 학생 강연 ([KOBA2024](https://kobashow.com/kor/index.asp#))
+ 한국 방송 미디어 공학회 2024년 하계학술대회 및 대학생 경진대회에 투고 및 구두 발표(6/28 예정)
+ 논문 : [Diffusion 모델 기반 깊이 추정을 통한 K-POP 직캠 인물 검출 성능 향상](https://github.com/Junoflows/AI_Fancam/blob/main/Diffusion%20%EB%AA%A8%EB%8D%B8%20%EA%B8%B0%EB%B0%98%20%EA%B9%8A%EC%9D%B4%20%EC%B6%94%EC%A0%95%EC%9D%84%20%ED%86%B5%ED%95%9C%20K-POP%20%EC%A7%81%EC%BA%A0%20%EC%9D%B8%EB%AC%BC%20%EA%B2%80%EC%B6%9C%20%EC%84%B1%EB%8A%A5%20%ED%96%A5%EC%83%81.pdf)
+ KOBA 발표 자료 : [깊이 추정을 통한 직캠 모델 성능 향상](https://github.com/Junoflows/AI_Fancam/blob/main/%EA%B9%8A%EC%9D%B4%20%EC%B6%94%EC%A0%95%EC%9D%84%20%ED%86%B5%ED%95%9C%20%EC%A7%81%EC%BA%A0%20%EB%AA%A8%EB%8D%B8%20%EC%84%B1%EB%8A%A5%20%ED%96%A5%EC%83%81.pdf)

## 요약
현재 많은 음악 방송에서는 아이돌 무대의 멤버별 직캠 영상을 제작하기 위해 많은 장비와 인력을 소비하고 있다. 
이러한 리소스 효율화를 위해 객체 검출 모델을 이용한 직캠 영상 제작 연구가 진행되고 있지만, 무대 위 동선이 겹치는
상황에서는 가려진 객체를 모두 검출하는 데 어려움이 있다. 본 논문에서는 겹친 데이터 중 가장 앞에 있는 데이터를
보존하기 위해 Diffusion 기반의 깊이 추정 모델을 사용하여 2D 이미지에서 깊이 정보를 추출하고, 이를 통해 객체 검출
모델의 성능을 향상시키는 방법을 제안한다. 추가 실험으로 학습 시 데이터 증강 방식으로 Grayscale 을 적용하여 모델의
일반화 성능을 높이는 방법도 제안한다. 제안한 방법을 통해 mAP 를 0.472 에서 0.512 로 향상시켰다. 

## 기존 문제점
![image](https://github.com/Junoflows/AI_Fancam/assets/108385417/8fd7182c-48be-4f6d-8fbb-2b02dd96c136)
+ 무대에 맴버가 겹치는 경우 
