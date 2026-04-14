# LoRa 페이로드 심볼 복조 실험 코드

이 저장소는 **LoRa 수신 과정 전체를 구현한 코드가 아니라, coarse synchronization 이후의 payload symbol demodulation 구간**을 대상으로 한다.

즉, 이 코드는 다음과 같은 질문에 답하기 위해 작성되었다.

- 매우 낮은 SNR 환경에서 기본적인 LoRa 복조기만 사용할 때 어떤 한계가 생기는가?
- 기존 복조기를 완전히 버리지 않고, 신뢰도가 낮은 경우에만 CNN을 보조적으로 사용하면 성능을 개선할 수 있는가?
- CNN을 모든 심볼에 항상 적용하지 않고도 의미 있는 성능 이득을 얻을 수 있는가?

## 1. 코드가 다루는 범위

이 저장소가 직접 다루는 범위는 다음과 같다.

- LoRa 심볼 생성
- 잔여 CFO, timing offset, multipath, phase noise, tone interference, AWGN이 포함된 수신 신호 생성
- 기본 복조기와 CNN 기반 복조기 비교
- confidence 기반 하이브리드 선택 정책 보정
- SER / PER / CNN 사용률 / 지연시간 측정

이 저장소가 직접 다루지 않는 범위는 다음과 같다.

- preamble detection
- packet synchronization loop 전체
- FEC / CRC / full frame decoding
- 실제 SDR 장비 기반 실측 데이터 수집
- 임베디드 배치나 하드웨어 가속기 수준의 최적화

## 2. 수신기 구성

실험에서는 네 가지 수신 경로를 비교한다.

### 2.1 Default LoRa

가장 기본적인 복조 경로이다.

- 수신 신호에 downchirp를 곱해 dechirp 수행
- FFT 수행
- grouped-bin 에너지로 심볼 결정

### 2.2 Enhanced LoRa

기본 복조기를 강화한 기준선이다.

- 여러 CFO 가설
- 여러 timing offset 가설
- 각 가설에서 dechirp + FFT 수행
- 가장 유리한 에너지를 주는 결과를 선택

### 2.3 Full CNN

모든 심볼에 대해 CNN을 적용하는 경로이다.

- 다중 CFO / timing hypothesis feature bank 생성
- 생성된 2채널 특징(real / imag)을 2D CNN에 입력
- CNN이 직접 심볼 클래스를 분류

### 2.4 Hybrid CNN

기본 복조기와 CNN을 함께 사용하는 경로이다.

- 먼저 Default LoRa 결과와 confidence를 계산
- confidence가 충분히 높으면 Default LoRa 결과 사용
- confidence가 낮으면 CNN 결과 사용

## 3. 채널 모델

이 코드는 단순 AWGN만 넣는 환경이 아니라, 더 복잡한 impairment를 함께 포함한다.

각 패킷은 payload 심볼 전체에 대해 하나의 공통 채널 상태를 유지하며, 다음 요소들이 포함될 수 있다.

- residual CFO
- integer timing offset
- fractional timing offset
- multipath
- carrier phase offset
- phase noise
- narrowband tone interference
- AWGN

평가용 채널은 두 가지 계열로 나뉜다.

### 3.1 seen_eval

학습 때 사용한 분포와 유사한 채널이다.

### 3.2 unseen_eval

학습보다 더 가혹한 분포를 사용한다.

- 더 큰 CFO / timing spread
- 더 많은 경로 수
- 더 강한 multipath
- 더 큰 phase noise
- 더 높은 interference 확률

## 4. 실험 프로파일

기본 설정은 서로 다른 LoRa 동작점에 대해 별도의 모델을 학습한다.

- `sf7_bw125`
- `sf8_bw125`
- `sf9_bw250`

각 프로파일은 다음 파라미터를 가진다.

- spreading factor
- bandwidth
- sampling rate
- 필요하면 프로파일별 학습 설정 오버라이드
- 필요하면 프로파일별 모델 크기 오버라이드

## 5. 측정 지표

코드는 다음 지표를 저장한다.

- `SER`: symbol error rate
- `PER`: packet error rate
- `CNN utilization`: 하이브리드 경로에서 CNN이 실제로 호출된 비율
- `latency`: 각 수신기 경로의 상대적인 추론 시간
- `parameter count`: CNN 학습 파라미터 수

CSV 결과는 다음 위치에 저장된다.

- `csv/experiment_summary.csv`
- `csv/latency_summary.csv`

그래프는 다음 위치에 저장된다.

- `graph/`

학습된 최적 모델은 다음 위치에 저장된다.

- `artifacts/weights/`
- `artifacts/checkpoints/`

## 6. 파일별 설명

### `config.py`

전체 실험 설정을 모아 두는 파일이다.

- 수신기 프로파일
- feature bank 설정
- 학습 설정
- 실험 설정
- 하이브리드 정책 설정
- 벤치마크 설정
- 채널 프로파일

### `simulator.py`

LoRa 심볼 생성과 채널 impairment 주입을 담당한다.

- upchirp / downchirp 생성
- multipath 적용
- timing shift 적용
- CFO 적용
- phase noise / interference 추가
- AWGN 추가
- 기본 복조기 연산
- multi-hypothesis feature bank 추출

### `dataset.py`

학습용 / 검증용 / 평가용 데이터셋을 만든다.

- 온라인 학습용 파라미터 샘플 dataset
- 고정된 waveform validation dataset
- SNR별 고정 waveform test dataset
- recorded IQ `.npz` 로더

### `models.py`

2D CNN 기반 복조 모델을 정의한다.

### `training.py`

학습 루프를 담당한다.

- 온라인 채널 샘플링
- feature extraction
- CNN 학습
- validation loss 추적
- best weights / checkpoint 저장

### `evaluation.py`

평가와 정책 보정을 담당한다.

- confidence 계산
- global threshold policy 보정
- confidence-bin policy 보정
- SER / PER 계산
- receiver latency 측정

### `main.py`

실험 전체를 orchestration하는 진입점이다.

- 프로파일별 설정 병합
- 데이터셋 생성
- 모델 학습
- calibration
- seen / unseen 평가
- CSV 저장
- 그래프 저장

### `utils.py`

여러 파일에서 공통으로 쓰는 유틸리티 모음이다.

- seed 고정
- CFO 범위 계산
- 파라미터 수 계산
- nested config 병합
- benchmark용 타이머

### `colab_run.py`

Google Colab에서 메모리와 실행 시간을 고려해 실험을 돌릴 수 있도록 preset을 적용하는 실행 파일이다.

## 7. 실행 방법

필수 패키지 설치:

```bash
pip install torch numpy pandas matplotlib
```

기본 실행:

```bash
python main.py
```

Colab용 preset 실행:

```bash
python colab_run.py --mode sf7
python colab_run.py --mode sf8
python colab_run.py --mode sf9
```

추가 override 예시:

```bash
python colab_run.py --mode sf7 --epochs 10 --train-samples 20000
```

## 8. 결과 해석 시 주의할 점

### 8.1 `0 dB`가 무조건 `SER = 0`을 의미하지는 않는다

이 코드의 SNR은 AWGN만 따지는 값이지만, 채널에는 그 외 impairment가 함께 포함될 수 있다.

즉 `0 dB`라고 해도

- timing mismatch
- multipath
- phase noise
- interference

때문에 오류가 남을 수 있다.

### 8.2 unseen 채널은 단조 감소하지 않을 수 있다

SNR이 증가해도 unseen 채널에서 SER가 완전히 단조롭게 내려가지 않을 수 있다.

이유는 다음과 같다.

- impairment-limited 환경일 수 있음
- SNR별 평가셋이 독립적으로 생성됨
- seed 수가 적으면 곡선이 더 흔들릴 수 있음

### 8.3 high-SNR 구간의 작은 스파이크는 표본 수 영향일 수 있다

특히 Colab처럼 평가셋을 줄인 설정에서는 SNR별 샘플 수가 적어서 작은 오차 하나가 SER 곡선에 눈에 띄게 반영될 수 있다.

## 9. recorded IQ 데이터 사용

외부 측정 데이터를 사용하려면 `.npz` 파일에 다음 항목이 들어 있어야 한다.

- `labels`
- `rx`
  또는
- `rx_real`, `rx_imag`

이 로더는 `dataset.py`의 `load_recorded_waveform_dataset(...)`에 구현되어 있다.

## 10. 추가 확장 방향

이 코드를 더 확장하려면 보통 다음 순서로 진행하면 된다.

1. 실측 IQ 데이터 연결
2. 더 정교한 synchronization 전처리 추가
3. concurrent LoRa interference 모델 추가
4. 프로파일별 학습 및 평가 반복
5. seed 반복을 통한 평균 / 표준편차 정리
6. 필요하면 full frame decoding 방향으로 확장
