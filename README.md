# LoRa Hybrid Receiver for Ultra-Low SNR

표준 LoRa `dechirp + FFT` 복조기의 **ultra-low SNR 취약 구간**을 신경망으로 보완하기 위한 연구용 코드이다.

이 저장소는 다음 질문에 답하기 위해 설계되었다.

- 기존 LoRa 복조기는 매우 낮은 SNR에서 어디서부터 흔들리는가
- 그 취약 구간만 신경망으로 보완하면 성능을 높일 수 있는가
- 항상 CNN을 쓰지 않고, **필요한 경우에만 선택적으로 호출**하면 계산량을 줄일 수 있는가

핵심 구조는 다음과 같다.

1. **Classical Receiver**
   - `dechirp + FFT`
   - grouped-bin 기반 점수 계산
   - top1 / top2 기반 confidence 산출

2. **Neural Receiver**
   - CFO / Timing Offset 다중 가설을 적용한 2차원 특징맵 생성
   - local frequency patch를 포함한 2D CNN 분류

3. **Hybrid Policy**
   - classical confidence가 낮을 때만 CNN 결과를 사용
   - 고정 threshold와 adaptive threshold를 모두 비교 평가

---

## 1. 프로젝트 개요

일반적인 LoRa 복조는 dechirp 후 FFT peak를 보는 방식으로 동작한다. 이 방법은 계산량이 작고 직관적이지만, SNR이 매우 낮고 CFO나 multipath가 존재하면 peak가 흐려져 오검출 가능성이 커진다.

이 프로젝트는 이 문제를 다음 방식으로 해결한다.

- 기존 LoRa 복조기를 **버리지 않는다**
- 먼저 classical receiver로 심볼을 추정한다
- classical receiver의 confidence를 계산한다
- confidence가 낮은 구간에서만 CNN을 호출한다
- 결과적으로 **성능과 계산량 사이의 trade-off**를 분석한다

즉, 이 코드는 end-to-end black box 수신기가 아니라, **기존 복조기를 유지한 채 취약 구간만 보조하는 hybrid receiver**를 구현한다.

---

## 2. 현재 코드의 핵심 아이디어

### 2.1 Classical Baseline
수신 신호에 downchirp를 곱해 dechirp를 수행한 뒤 FFT를 적용한다.  
각 심볼 중심 bin 주변의 에너지를 합산하여 grouped-bin score를 계산하고, 가장 큰 score를 갖는 심볼을 baseline prediction으로 사용한다.

### 2.2 Multi-Hypothesis 2D Feature Bank
낮은 SNR 환경에서는 단일 dechirp + FFT 결과만으로는 peak 구조를 제대로 읽기 어렵다.  
그래서 이 코드에서는 다음 가설을 동시에 고려한다.

- CFO hypothesis 여러 개
- Timing Offset hypothesis 여러 개
- 각 심볼 중심 주변의 local frequency patch

이렇게 만들어진 결과를 하나의 2차원 특징맵으로 쌓아 CNN 입력으로 사용한다.

입력 텐서의 기본 형태는 다음과 같다.

- `[Batch, 2, Num_Hypotheses, Num_Bins]`

여기서
- `2`는 실수부 / 허수부 채널
- `Num_Hypotheses = cfo_steps * to_steps`
- `Num_Bins = (2 ** sf) * patch_size`

를 의미한다.

### 2.3 Hybrid Decision
classical grouped energy로부터 confidence를 계산한다.  
현재 지원하는 confidence는 다음과 같다.

- `ratio`
- `norm_margin`
- `entropy`

기본값은 `ratio`이다.

confidence가 충분히 높으면 classical 결과를 그대로 사용한다.  
confidence가 낮으면 CNN 결과를 사용한다.

즉 최종 결정은 다음 구조를 따른다.

- easy case → classical
- hard case → CNN

---

## 3. 파일 구조

```text
.
├── config.py
├── dataset.py
├── evaluation.py
├── main.py
├── models.py
├── simulator.py
├── training.py
├── utils.py
└── README.md
```

---

## 4. 파일별 설명

### 4.1 `config.py`
실험에 사용하는 주요 하이퍼파라미터를 중앙에서 관리한다.

포함 내용:
- 물리 계층 설정
  - `sf`
  - `bw`
  - `fs`
- 다중 가설 설정
  - `max_cfo_bins`
  - `patch_size`
  - `cfo_steps`
  - `to_steps`
  - `max_to_samples`
- 학습 설정
  - `train_batch_size`
  - `eval_batch_size`
  - `num_epochs`
  - `learning_rate`
  - `packet_size`
- 실험 설정
  - `seeds`
  - `test_snrs`
  - `calib_samples`
  - `test_samples`
  - `train_samples`

이 파일을 사용하면 여러 소스 파일에 흩어진 숫자를 한 곳에서 바꿀 수 있어 실험 관리가 쉬워진다.

### 4.2 `dataset.py`
학습용 파라미터 데이터셋과, 고정 validation/calibration/test 데이터셋을 생성한다.

#### `OnlineParametersDataset`
실제 파형을 저장하지 않고, 샘플마다 `label`, `SNR`, `CFO`만 생성한다.  
실제 수신 파형은 학습 루프에서 시뮬레이터가 온라인으로 생성한다.

#### `create_fixed_feature_dataset(...)`
고정 validation dataset을 만든다.

- label / SNR / CFO를 고정 생성한다
- 실제 파형을 생성한다
- 다중 가설 2D 특징맵으로 변환한다
- `TensorDataset(labels, features)` 형태로 저장한다

#### `create_fixed_waveform_dataset(...)`
calibration/test용 고정 dataset을 만든다.

- SNR별로 별도의 파형 세트를 생성한다
- feature가 아니라 `rx_signals` 자체를 저장한다

이유:
- baseline grouped-bin 계산 필요
- CNN feature bank 재생성 필요
- hybrid policy 평가 필요

### 4.3 `simulator.py`
프로젝트의 핵심 시뮬레이터이다.

역할:
1. 정답 심볼로부터 LoRa 송신 파형 생성
2. multipath 적용
3. CFO 적용
4. AWGN 추가
5. dechirp + FFT 특징 추출
6. grouped-bin baseline 계산
7. 다중 가설 2D feature bank 생성

#### 주요 함수

##### `generate_batch(...)`
입력:
- `labels`
- `snrs_db`
- `cfos_hz`

출력:
- multipath, CFO, noise가 적용된 복소 수신 파형

##### `extract_features(...)`
단일 가설 dechirp + FFT 특징을 만든다.  
형태는 `[Batch, 2, N]`이다.

##### `baseline_grouped_bin(...)`
classical grouped-bin baseline를 계산한다.  
각 심볼 중심 주변 에너지를 합산하여 점수를 만든다.

##### `generate_hypothesis_grid(...)`
CFO / timing offset 가설 격자를 만든다.

##### `extract_multi_hypothesis_bank(...)`
이 프로젝트의 핵심 함수이다.

처리 흐름:
1. timing offset 가설별 신호 시프트
2. CFO 가설별 보정 적용
3. dechirp 수행
4. FFT 수행
5. 각 심볼 중심 주변 patch 추출
6. 모든 결과를 2차원 feature bank로 결합

### 4.4 `models.py`
다중 가설 2차원 특징맵을 입력으로 받아 LoRa 심볼을 분류하는 2D CNN 모델을 정의한다.

#### `Hypothesis2DCNN`
입력 형태:
- `[Batch, 2, Num_Hypotheses, Num_Bins]`

구성:
- 2D convolution 블록 4개
- BatchNorm
- ReLU
- MaxPool
- Flatten
- Fully Connected classifier

설계 의도:
- 가설 축 방향 패턴을 본다
- 주파수 축 patch 구조를 본다
- 둘을 동시에 읽어 ultra-low SNR에서 더 robust한 심볼 분류를 수행한다

### 4.5 `training.py`
모델 학습을 담당한다.

#### `train_online_model(...)`
특징:
- 학습용 파형을 미리 저장하지 않는다
- 매 배치마다 simulator가 새로운 파형을 생성한다
- 다중 가설 2D feature bank를 만들어 CNN에 넣는다
- validation loss가 가장 낮은 시점의 모델을 최종 복원한다

### 4.6 `evaluation.py`
hybrid 정책 보정과 최종 평가를 담당한다.

#### `get_confidence(...)`
classical grouped energy로부터 confidence를 계산한다.

#### `calibrate_adaptive_policy_joint(...)`
SNR별 adaptive threshold를 calibration dataset으로 자동 탐색한다.

목표:
- SER / PER이 너무 나빠지지 않도록 제한한다
- 그 조건 안에서 CNN utilization이 가장 낮은 threshold를 선택한다

#### `run_evaluation(...)`
fixed 또는 adaptive policy로 최종 성능을 평가한다.

반환 정보:
- `ser_g` : classical baseline SER
- `ser_c` : full CNN SER
- `ser_h` : hybrid SER
- `per_g` : classical baseline PER
- `per_c` : full CNN PER
- `per_h` : hybrid PER
- `util`  : CNN utilization
- `th`    : 사용된 threshold

### 4.7 `main.py`
전체 파이프라인을 실행하는 메인 스크립트이다.

실행 흐름:
1. seed별 반복 실험
2. simulator / model 생성
3. validation / calibration / seen / unseen dataset 생성
4. 온라인 학습
5. adaptive policy 보정
6. fixed / adaptive, seen / unseen 평가
7. CSV 저장
8. 그래프 생성

또한 다음 두 종류의 시각화를 생성한다.

#### `plot_summary(...)`
- conventional LoRa
- full CNN
- hybrid
- utilization

을 한 그래프에 보여준다.

#### `plot_ablation(...)`
- fixed hybrid
- adaptive hybrid

를 직접 비교한다.

### 4.8 `utils.py`
재현성을 위한 seed 고정 함수가 들어 있다.

#### `set_seed(...)`
다음 난수 흐름을 고정한다.
- NumPy
- PyTorch
- CUDA
- Python hash seed
- cuDNN deterministic 옵션

---

## 5. 전체 파이프라인 요약

이 프로젝트의 전체 흐름은 아래와 같다.

### 단계 1. 파라미터 생성
학습용 데이터셋은 label / SNR / CFO만 제공한다.

### 단계 2. 온라인 파형 생성
시뮬레이터가 GPU에서 즉석으로 LoRa 파형을 만든다.

### 단계 3. Classical Receiver 계산
dechirp + FFT + grouped-bin score를 계산한다.

### 단계 4. Multi-Hypothesis Feature Bank 생성
CFO / timing offset 가설과 local patch를 사용해 2D 입력을 만든다.

### 단계 5. CNN 분류
2D CNN이 심볼을 분류한다.

### 단계 6. Confidence 기반 Hybrid 선택
classical confidence가 높으면 baseline을 사용한다.  
낮으면 CNN 결과를 사용한다.

### 단계 7. SER / PER / Utilization 분석
- baseline
- full CNN
- hybrid
- fixed vs adaptive

를 비교한다.

---

## 6. 설치 방법

### 권장 환경
- Python 3.10+
- PyTorch
- NumPy
- pandas
- matplotlib

### 예시 설치
```bash
pip install torch numpy pandas matplotlib
```

CUDA 사용 환경이면 GPU 버전에 맞는 PyTorch 설치가 필요하다.

---

## 7. 실행 방법

기본 실행:

```bash
python main.py
```

실행 시 다음 작업이 순서대로 수행된다.

- seed 반복
- 모델 학습
- adaptive policy calibration
- seen / unseen evaluation
- 결과 CSV 저장
- summary plot 저장
- ablation plot 저장

---

## 8. 출력 파일

실행이 끝나면 보통 다음 파일들이 생성된다.

- `experiment_summary.csv`
- `experiment_main_seen.png`
- `experiment_main_unseen.png`
- `experiment_ablation_seen.png`
- `experiment_ablation_unseen.png`

### CSV 주요 컬럼
- `type`
  - `fixed_seen`
  - `adapt_seen`
  - `fixed_unseen`
  - `adapt_unseen`
- `snr`
- `ser_g_mean`, `ser_c_mean`, `ser_h_mean`
- `per_g_mean`, `per_c_mean`, `per_h_mean`
- `util_mean`
- `th_mean`
- `*_std`
- `n_runs`

---

## 9. 주요 설정값 설명

### `sf`
Spreading Factor이다.  
심볼 개수는 `2 ** sf`로 결정된다.

### `max_cfo_bins`
허용할 CFO 범위를 bin 단위로 지정한다.

### `patch_size`
각 심볼 중심 주변 몇 개 bin을 함께 볼지 결정한다.

### `cfo_steps`, `to_steps`
CFO / Timing Offset 가설 개수이다.  
이 값이 커질수록 특징맵이 커지고 계산량도 증가한다.

### `packet_size`
PER 계산을 위해 몇 개 심볼을 한 패킷으로 묶을지 정한다.

### `seeds`
반복 실험 개수이다.  
평균과 표준편차 계산에 사용된다.

---

## 10. 결과 해석 방법

### `ser_g`
classical grouped-bin baseline의 SER이다.

### `ser_c`
full CNN을 항상 사용했을 때의 SER이다.

### `ser_h`
hybrid policy를 사용했을 때의 SER이다.

### `util`
전체 샘플 중 실제로 CNN이 호출된 비율이다.

### 좋은 결과의 의미
보통 아래 형태가 이상적이다.

- `ser_c < ser_g`
- `ser_h`가 `ser_c`에 가깝다
- `util`은 full CNN 대비 낮다

즉 CNN을 항상 쓰지 않아도, 어려운 구간만 선택적으로 보완하여 성능을 유지한다는 뜻이다.

---

## 11. 이 프로젝트의 강점

- 기존 LoRa 수신기를 버리지 않는다
- classical receiver와 neural receiver를 함께 사용한다
- ultra-low SNR 취약 구간만 집중적으로 보완한다
- 성능뿐 아니라 CNN 호출률까지 함께 평가한다
- fixed policy와 adaptive policy를 모두 비교한다
- seen / unseen harsher channel을 분리 평가한다

---

## 12. 한계와 주의점

이 프로젝트는 연구용 시뮬레이션 코드이다.  
따라서 다음 한계가 있다.

- 실제 SDR 하드웨어 입력을 직접 처리하지 않는다
- 완전한 LoRa PHY/MAC 스택 전체를 구현한 것은 아니다
- packet_size 기반 PER은 연구용 surrogate metric이다
- Python / PyTorch 기반이라 임베디드 배치용 코드는 아니다

즉, 이 저장소는 **실제 수신기 아이디어를 검증하는 연구용 프로토타입**에 가깝다.

---

## 13. 실용적 의미

이 연구는 기존 LoRa 복조기를 완전히 대체하려는 것이 아니다.  
대신 다음 상황을 겨냥한다.

- ultra-low SNR에서 classical receiver가 애매해지는 구간
- 게이트웨이 또는 연산 여유가 있는 수신기
- 선택적으로 neural refinement를 붙이고 싶은 경우

따라서 가장 자연스러운 적용 대상은 다음과 같다.

- LoRa gateway
- SDR 기반 수신기
- PC / GPU 기반 실험 수신기
- 고신뢰 복조가 필요한 서버 측 receiver

---

## 14. 향후 확장 아이디어

- 실제 SDR capture 데이터 평가
- 더 작은 모델 경량화
- learned gate 도입
- multi-SF / multi-BW 확장
- real-time inference profiling
- ONNX / TensorRT 배포 최적화

---

## 15. 저장소 사용 목적

이 저장소는 다음 목적에 적합하다.

- LoRa hybrid receiver 구조 학습
- ultra-low SNR 보조 복조 연구

---

## 16. 실행 전 체크리스트

실행 전 아래를 확인하면 좋다.

- CUDA가 정상 인식되는가
- `config.py` 값이 현재 실험 목적과 맞는가
- validation / calibration / test 샘플 수가 너무 크지 않은가
- GPU 메모리와 시스템 RAM이 충분한가
- 출력 CSV / 그래프 파일명이 기존 결과를 덮어쓰지 않는가

---

## 17. 한 줄 요약

이 프로젝트는 **표준 LoRa dechirp + FFT 복조기의 ultra-low SNR 취약 구간을, 다중 가설 2D CNN과 adaptive hybrid policy로 선택적으로 보완하는 연구용 receiver framework**이다.
