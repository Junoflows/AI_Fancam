# 깊이 추정을 통한 AI 직캠 인물 검출 성능 향상
서울시립대와 KBS 미디어 기술 연구소와 함께 진행한 프로젝트로, 해당 주제로 작성한 논문입니다.
+ KOBA 2024 KBS 공개 AI 세미나 학생 강연 ([KOBA2024](https://kobashow.com/kor/index.asp#))
+ 한국 방송 미디어 공학회 2024년 하계학술대회 및 대학생 경진대회에 투고 및 구두 발표(6/28 예정)
+ 논문 : [Diffusion 모델 기반 깊이 추정을 통한 K-POP 직캠 인물 검출 성능 향상](https://github.com/Junoflows/AI_Fancam/blob/main/Diffusion%20%EB%AA%A8%EB%8D%B8%20%EA%B8%B0%EB%B0%98%20%EA%B9%8A%EC%9D%B4%20%EC%B6%94%EC%A0%95%EC%9D%84%20%ED%86%B5%ED%95%9C%20K-POP%20%EC%A7%81%EC%BA%A0%20%EC%9D%B8%EB%AC%BC%20%EA%B2%80%EC%B6%9C%20%EC%84%B1%EB%8A%A5%20%ED%96%A5%EC%83%81.pdf)
+ KOBA 발표 : [깊이 추정을 통한 직캠 모델 성능 향상](https://github.com/Junoflows/AI_Fancam/blob/main/%EA%B9%8A%EC%9D%B4%20%EC%B6%94%EC%A0%95%EC%9D%84%20%ED%86%B5%ED%95%9C%20%EC%A7%81%EC%BA%A0%20%EB%AA%A8%EB%8D%B8%20%EC%84%B1%EB%8A%A5%20%ED%96%A5%EC%83%81.pdf)
+ [중앙일보 뉴스](https://www.joongang.co.kr/article/25255627#home)
<img src = 'https://pds.joongang.co.kr/news/component/htmlphoto_mmdata/202406/11/1dba0302-339b-4f40-a5d1-5407c964b955.jpg' width = 300 height = 200>

## 요약
현재 많은 음악 방송에서는 아이돌 무대의 멤버별 직캠 영상을 제작하기 위해 많은 장비와 인력을 소비하고 있다. 
이러한 리소스 효율화를 위해 객체 검출 모델을 이용한 직캠 영상 제작 연구가 진행되고 있지만, 무대 위 동선이 겹치는
상황에서는 가려진 객체를 모두 검출하는 데 어려움이 있다. 본 논문에서는 겹친 데이터 중 가장 앞에 있는 데이터를
보존하기 위해 Diffusion 기반의 깊이 추정 모델을 사용하여 2D 이미지에서 깊이 정보를 추출하고, 이를 통해 객체 검출
모델의 성능을 향상시키는 방법을 제안한다. 추가 실험으로 학습 시 데이터 증강 방식으로 Grayscale 을 적용하여 모델의
일반화 성능을 높이는 방법도 제안한다. 제안한 방법을 통해 mAP 를 0.472 에서 0.512 로 향상시켰다. 

## AI 직캠이란
<img src = 'https://github.com/Junoflows/AI_Fancam/assets/108385417/684af815-c82d-47e8-9066-86193c69971b' width = 600 height = 400 >

+ 초고화질 카메라로 무대 영상을 찍은 뒤 AI 기술을 활용하여 맴버별 직캠 영상을 도출하는 작업
+ 카메라 비용, 카메라맨, 공간 등의 COST를 절약할 수 있음
+ 실제 KBS에서 AI 기술을 활용한 직캠을 유튜브에 업로드 중 ([KBS 뮤직뱅크 AI 직캠](https://youtube.com/playlist?list=PLK8rVA0_KzOcra8_HmOVIfd2vBoVGaKEE&si=OeLZvWnvimkW__Gb))
+ AI 직캠의 성능을 향상시키는 것이 연구 목적

## 기존 문제점
<img src = 'https://github.com/Junoflows/AI_Fancam/assets/108385417/b76d9bea-f486-4a2f-96de-2608b84653b8' width = 600 height = 400 >

+ 무대에 맴버가 겹치는 경우 두 박스를 모두 학습에 포함시키면 뒤의 박스는 훈련에 방해됨(직캠으로의 역할 X)

## 기존 방식의 해결 방안
<img src = 'https://github.com/Junoflows/AI_Fancam/assets/108385417/4415663a-736a-46bc-96f2-5de8916c9ff6' width = 600 height = 400 >

+ 유의미한 앞의 박스도 제거되어 정보의 손실 발생

## 해결 방안(핵심 아이디어)
<img src = 'https://github.com/Junoflows/AI_Fancam/assets/108385417/ff75ffdd-5887-40c9-beea-8efacc00235c' width = 600 height = 400 >

+ 깊이 추정 알고리즘을 도입하여 바운딩 박스의 깊이 정보를 파악하여 앞의 박스를 검출

## 연구 결과
<img src = 'https://github.com/Junoflows/AI_Fancam/assets/108385417/2e2befea-e41d-40cf-9178-ecf4c4c6cce1' width = 600 height = 400 >

+ 데이터 정제 과정에 깊이 추정 모델을 도입한 모델이 기존 방식보다 성능 향상을 이뤄냈음을 확인

## 시연 영상
https://drive.google.com/drive/folders/1AIUz5S5HUWdBxJQipuqRfO0VNr_lIeQx?usp=sharing

+ 기존 방식의 모델과 제안한 방식의 모델의 AI 직캠 시연 영상
+ 동선이 겹쳤을 때, 빠르게 이동하는 동선 등에서 제안한 방식의 AI 직캠이 기존의 방식보다 자연스럽게 이어지는 것을 확인함


## Open Source 및 모델
[OpenMMlab - mmedetection](https://github.com/open-mmlab)
+ detection을 위한 tool로써 open source 인 OpenMMlab의 mmdetection 사용
+ detection 모델은 2024.05.01 기준 SOTA를 달성한 Co-Detr 모델 사용
+ Depth Estimation 모델은 2023.12에 올라온 Marigold 모델 사용
  + 새로운 이미지에 대한 성능이 좋은 Zero-shot 능력이 뛰어난 모델
