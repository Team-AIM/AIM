# AIM
Project Name : Brain Tumor Segmentation using Label-specific SwinUNETR and Feature Fusion

<img width="784" height="443" alt="image" src="https://github.com/user-attachments/assets/54db6779-7eb6-42c8-8d66-d671836205e0" />

1. BraTS2025 Task2 Dataset 나누기 -> `train_5fold_data.json`, `test_data.json`
2. **`train_Label-specific_SwinUNETR.py`** 에서 라벨만 바꾸어 3번 학습 시행
    - input : train 5fold data (t1n, t1ce, t2w, t2f)
    - output : `SwinUNETR_et.pt`, `SwinUNETR_tc.pt`, `SwinUNETR_flair.pt`
3. 위에서 학습된 모델들을 **`test_SwinUNETR.ipynb`** 에서 inference & test
4. **`train_Feature_Fusion.py`** 에서 fusion하는 과정을 cnn으로 학습
    - input : `SwinUNETR_et.pt`에서 나온 확률맵, `SwinUNETR_tc.pt`에서 나온 확률맵, `SwinUNETR_flair.pt`에서 나온 확률맵
    - output : `Best_Fusion.pth`
5. **`test_Feature_Fusion_SwinUNETR.ipynb`** 로 최종 결과 test
