"""학습된 수신기들을 평가하고, 하이브리드 정책을 보정하는 파일이다.

이 파일은 다음 작업을 담당한다.

- 기본 복조기 score로부터 confidence 계산
- calibration 데이터에서 하이브리드 정책 보정
- seen / unseen 데이터셋에 대한 SER / PER 계산
- 각 수신기 경로의 상대적 지연시간 측정
"""

from typing import Dict

import torch
import torch.nn.functional as F

from config import CFG
from utils import benchmark_callable, get_max_cfo_hz


def get_confidence(grouped_energy: torch.Tensor, conf_type: str = "ratio") -> torch.Tensor:
    """기본 복조기 출력 에너지로부터 confidence를 계산한다.

    confidence는 하이브리드 정책이 CNN을 호출할지 말지를 결정하는 기준이 된다.
    """

    top2_values, _ = torch.topk(grouped_energy, 2, dim=1)
    t1, t2 = top2_values[:, 0], top2_values[:, 1]

    if conf_type == "ratio":
        return t1 / (t2 + 1e-9)

    if conf_type == "norm_margin":
        return (t1 - t2) / (t1 + 1e-9)

    if conf_type == "entropy":
        probabilities = F.softmax(grouped_energy, dim=1)
        return -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=1)

    raise ValueError(f"Unsupported confidence type: {conf_type}")


def _compute_packet_error_rate(labels: torch.Tensor, predictions: torch.Tensor, payload_symbols: int) -> float:
    """패킷 안 심볼 중 하나라도 틀리면 해당 패킷을 오류로 보는 PER를 계산한다."""

    if labels.numel() != predictions.numel():
        raise ValueError("labels and predictions must have the same number of samples.")
    if labels.numel() % payload_symbols != 0:
        raise ValueError("The number of samples must be divisible by payload_symbols.")
    num_packets = labels.numel() // payload_symbols
    labels_packet = labels.view(num_packets, payload_symbols)
    preds_packet = predictions.view(num_packets, payload_symbols)
    return (torch.any(labels_packet != preds_packet, dim=1)).float().mean().item()


def _compute_sample_error_rate(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    """심볼 단위 오류율인 SER를 계산한다."""

    if labels.numel() != predictions.numel():
        raise ValueError("labels and predictions must have the same number of samples.")
    return 1.0 - (labels == predictions).float().mean().item()


def _materialize_policy(policy: Dict, device: torch.device, dtype: torch.dtype) -> Dict:
    """파이썬 dict로 저장된 정책을 텐서 기반 실행 형태로 변환한다."""

    if policy["mode"] == "threshold":
        return policy

    if policy["mode"] == "bin":
        return {
            "mode": "bin",
            "edges": torch.tensor(policy["edges"], device=device, dtype=dtype),
            "use_cnn_by_bin": torch.tensor(policy["use_cnn_by_bin"], device=device, dtype=torch.bool),
        }

    raise ValueError(f"Unsupported policy mode: {policy['mode']}")


def _policy_mask(confidence: torch.Tensor, policy: Dict, conf_type: str) -> torch.Tensor:
    """각 샘플에서 CNN을 사용할지 여부를 bool mask로 반환한다."""

    if policy["mode"] == "threshold":
        threshold = policy["threshold"]
        return (confidence > threshold) if conf_type == "entropy" else (confidence < threshold)

    if policy["mode"] == "bin":
        bin_ids = torch.bucketize(confidence, policy["edges"][1:-1], right=False)
        return policy["use_cnn_by_bin"][bin_ids]

    raise ValueError(f"Unsupported policy mode: {policy['mode']}")


def _flatten_outputs(records_by_snr: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """SNR별 레코드를 하나의 큰 텐서 묶음으로 합친다."""

    merged = {}
    keys = next(iter(records_by_snr.values())).keys()
    for key in keys:
        merged[key] = torch.cat([records_by_snr[snr][key] for snr in sorted(records_by_snr.keys())], dim=0)
    return merged


def _validate_record(record: Dict[str, torch.Tensor]) -> None:
    """레코드 안의 예측 길이가 모두 동일한지 검증한다."""

    base_length = record["labels"].numel()
    for key in ("pred_single", "pred_cnn", "confidence"):
        if record[key].numel() != base_length:
            raise ValueError(f"`{key}` has {record[key].numel()} samples but expected {base_length}.")


def collect_receiver_outputs(
    model,
    simulator,
    dataset_dict,
    channel_profile,
    feature_cfg=None,
    eval_batch_size=None,
    hybrid_cfg=None,
):
    """데이터셋 전체에 대해 각 수신기 경로의 출력을 수집한다.

    반환값은 SNR별로 다음 항목을 가진 딕셔너리다.

    - labels
    - Default LoRa 예측
    - Full CNN 예측
    - Default LoRa confidence
    """

    # 별도 설정을 넘기지 않으면 전역 기본 설정을 사용한다.
    feature_cfg = CFG["feature_bank"] if feature_cfg is None else feature_cfg
    eval_batch_size = CFG["training"]["eval_batch_size"] if eval_batch_size is None else eval_batch_size
    hybrid_cfg = CFG["hybrid"] if hybrid_cfg is None else hybrid_cfg
    conf_type = hybrid_cfg["confidence_type"]

    # 현재 채널 프로파일과 feature bank 설정에 맞춰 hypothesis helper를 준비한다.
    resolved_profile = simulator.resolve_channel_profile(channel_profile)
    max_cfo_hz = get_max_cfo_hz(simulator, channel_profile)
    cfo_grid, to_grid = simulator.generate_hypothesis_grid(
        max_cfo_hz,
        resolved_profile["max_to_samples"],
        feature_cfg["cfo_steps"],
        feature_cfg["to_steps"],
    )
    helper = simulator.prepare_hypothesis_helper(
        cfo_grid,
        to_grid,
        feature_cfg["patch_size"],
    )

    outputs = {}
    model.eval()
    with torch.inference_mode():
        # dataset_dict는 {snr: TensorDataset} 형태이므로 SNR별로 따로 결과를 모은다.
        for snr in sorted(dataset_dict):
            labels, rx_signals = dataset_dict[snr].tensors
            labels = labels.to(simulator.device)
            rx_signals = rx_signals.to(simulator.device)

            pred_single = []
            pred_cnn = []
            confidence = []

            # 추론도 메모리를 고려해 배치 단위로 나누어 수행한다.
            for start in range(0, labels.size(0), eval_batch_size):
                end = min(start + eval_batch_size, labels.size(0))
                rx_batch = rx_signals[start:end]

                # Default LoRa 경로: dechirp + FFT + grouped-bin
                grouped_single, _ = simulator.baseline_grouped_bin(
                    rx_batch,
                    window_size=feature_cfg["baseline_window"],
                )
                pred_single.append(torch.argmax(grouped_single, dim=1))
                confidence.append(get_confidence(grouped_single, conf_type=conf_type))

                # Full CNN 입력용 hypothesis feature bank를 추출한다.
                features = simulator.extract_multi_hypothesis_bank(
                    rx_batch,
                    helper=helper,
                )
                pred_cnn.append(torch.argmax(model(features), dim=1))

            # GPU 메모리 점유를 줄이기 위해 결과는 CPU로 내려 저장한다.
            record = {
                "labels": labels.cpu(),
                "pred_single": torch.cat(pred_single).cpu(),
                "pred_cnn": torch.cat(pred_cnn).cpu(),
                "confidence": torch.cat(confidence).cpu(),
            }
            _validate_record(record)
            outputs[snr] = record

    return outputs


def calibrate_global_threshold_from_outputs(
    records_by_snr: Dict[int, Dict[str, torch.Tensor]],
    hybrid_cfg=None,
    payload_symbols=None,
):
    """단일 threshold 하나로 하이브리드 정책을 보정한다.

    목표는 Full CNN과 거의 비슷한 SER / PER를 유지하면서 CNN 사용률을 최소화하는 것이다.
    """

    hybrid_cfg = CFG["hybrid"] if hybrid_cfg is None else hybrid_cfg
    payload_symbols = CFG["experiment"]["payload_symbols"] if payload_symbols is None else payload_symbols
    conf_type = hybrid_cfg["confidence_type"]

    # 여러 SNR calibration 데이터를 한데 합쳐 threshold 하나를 고른다.
    records = _flatten_outputs(records_by_snr)
    labels = records["labels"]
    pred_single = records["pred_single"]
    pred_cnn = records["pred_cnn"]
    confidence = records["confidence"]

    ser_c = _compute_sample_error_rate(labels, pred_cnn)
    per_c = _compute_packet_error_rate(labels, pred_cnn, payload_symbols)
    target_ser = ser_c + hybrid_cfg["ser_tolerance"]
    target_per = per_c + hybrid_cfg["per_tolerance"]

    # confidence 분포의 분위수 기반 후보 threshold를 만든다.
    candidate_grid = torch.linspace(
        0.0,
        1.0,
        steps=hybrid_cfg["global_threshold_grid"],
        device=confidence.device,
    )
    thresholds = torch.unique(torch.quantile(confidence, candidate_grid))

    best_policy = None
    best_util = float("inf")
    fallback_policy = None
    fallback_penalty = float("inf")

    for threshold in thresholds:
        # threshold에 따라 CNN 사용 여부를 정하고,
        # 그때의 SER / PER / CNN 사용률을 계산한다.
        use_cnn = (confidence > threshold) if conf_type == "entropy" else (confidence < threshold)
        pred_hybrid = torch.where(use_cnn, pred_cnn, pred_single)

        ser_h = _compute_sample_error_rate(labels, pred_hybrid)
        per_h = _compute_packet_error_rate(labels, pred_hybrid, payload_symbols)
        util = use_cnn.float().mean().item() * 100.0

        if ser_h <= target_ser and per_h <= target_per and util < best_util:
            best_util = util
            best_policy = {"mode": "threshold", "threshold": float(threshold.item())}

        penalty = max(0.0, ser_h - target_ser) * 1000.0 + max(0.0, per_h - target_per) * 500.0 + util
        if penalty < fallback_penalty:
            fallback_penalty = penalty
            fallback_policy = {"mode": "threshold", "threshold": float(threshold.item())}

    # 목표 SER/PER를 만족하는 가장 싼 정책이 있으면 그것을,
    # 없으면 penalty가 가장 작은 대체 정책을 사용한다.
    return best_policy if best_policy is not None else fallback_policy


def calibrate_confidence_bin_policy_from_outputs(
    records_by_snr: Dict[int, Dict[str, torch.Tensor]],
    hybrid_cfg=None,
    payload_symbols=None,
):
    """confidence 구간별로 CNN 사용 여부를 정하는 bin policy를 보정한다."""

    hybrid_cfg = CFG["hybrid"] if hybrid_cfg is None else hybrid_cfg
    payload_symbols = CFG["experiment"]["payload_symbols"] if payload_symbols is None else payload_symbols
    conf_type = hybrid_cfg["confidence_type"]

    records = _flatten_outputs(records_by_snr)
    labels = records["labels"]
    pred_single = records["pred_single"]
    pred_cnn = records["pred_cnn"]
    confidence = records["confidence"]

    ser_c = _compute_sample_error_rate(labels, pred_cnn)
    per_c = _compute_packet_error_rate(labels, pred_cnn, payload_symbols)
    target_ser = ser_c + hybrid_cfg["ser_tolerance"]
    target_per = per_c + hybrid_cfg["per_tolerance"]

    # confidence를 분위수 기반 bin으로 나누어 구간별 정책을 만든다.
    quantiles = torch.linspace(
        0.0,
        1.0,
        steps=hybrid_cfg["confidence_bins"] + 1,
        device=confidence.device,
    )
    edges = torch.quantile(confidence, quantiles)
    edges[0] = torch.tensor(float("-inf"), device=confidence.device)
    edges[-1] = torch.tensor(float("inf"), device=confidence.device)

    bin_ids = torch.bucketize(confidence, edges[1:-1], right=False)
    num_bins = hybrid_cfg["confidence_bins"]

    bin_means = []
    for bin_idx in range(num_bins):
        mask = bin_ids == bin_idx
        if torch.any(mask):
            bin_means.append((bin_idx, float(confidence[mask].mean().item())))
        else:
            bin_means.append((bin_idx, float(edges[bin_idx + 1].item())))

    reverse = conf_type == "entropy"
    ordered_bins = [bin_idx for bin_idx, _ in sorted(bin_means, key=lambda item: item[1], reverse=reverse)]

    best_policy = None
    best_util = float("inf")
    fallback_policy = None
    fallback_penalty = float("inf")

    use_cnn_by_bin = torch.zeros(num_bins, dtype=torch.bool, device=confidence.device)
    # confidence가 낮은 쪽부터 차례로 CNN을 켰을 때의 성능을 평가한다.
    for cutoff in range(num_bins + 1):
        if cutoff > 0:
            use_cnn_by_bin[ordered_bins[cutoff - 1]] = True

        use_cnn = use_cnn_by_bin[bin_ids]
        pred_hybrid = torch.where(use_cnn, pred_cnn, pred_single)
        ser_h = _compute_sample_error_rate(labels, pred_hybrid)
        per_h = _compute_packet_error_rate(labels, pred_hybrid, payload_symbols)
        util = use_cnn.float().mean().item() * 100.0

        policy = {
            "mode": "bin",
            "edges": [float(value) for value in edges.tolist()],
            "use_cnn_by_bin": [bool(value) for value in use_cnn_by_bin.tolist()],
        }

        if ser_h <= target_ser and per_h <= target_per and util < best_util:
            best_util = util
            best_policy = policy.copy()

        penalty = max(0.0, ser_h - target_ser) * 1000.0 + max(0.0, per_h - target_per) * 500.0 + util
        if penalty < fallback_penalty:
            fallback_penalty = penalty
            fallback_policy = policy.copy()

    return best_policy if best_policy is not None else fallback_policy


def summarize_outputs(
    records_by_snr: Dict[int, Dict[str, torch.Tensor]],
    policy: Dict,
    hybrid_cfg=None,
    payload_symbols=None,
):
    """보정된 정책을 적용해 SNR별 SER / PER / utilization 통계를 계산한다."""

    hybrid_cfg = CFG["hybrid"] if hybrid_cfg is None else hybrid_cfg
    payload_symbols = CFG["experiment"]["payload_symbols"] if payload_symbols is None else payload_symbols
    conf_type = hybrid_cfg["confidence_type"]
    compiled_policy = None
    stats = {}

    for snr, record in records_by_snr.items():
        _validate_record(record)
        labels = record["labels"]
        pred_single = record["pred_single"]
        pred_cnn = record["pred_cnn"]
        confidence = record["confidence"]

        if compiled_policy is None:
            compiled_policy = _materialize_policy(policy, confidence.device, confidence.dtype)

        # 실제 하이브리드 예측은 "CNN을 쓸 샘플만 CNN 결과 사용, 나머지는 기본 복조기 사용"으로 만든다.
        use_cnn = _policy_mask(confidence, compiled_policy, conf_type=conf_type)
        pred_hybrid = torch.where(use_cnn, pred_cnn, pred_single)

        stats[snr] = {
            "ser_single": _compute_sample_error_rate(labels, pred_single),
            "ser_c": _compute_sample_error_rate(labels, pred_cnn),
            "ser_h": _compute_sample_error_rate(labels, pred_hybrid),
            "per_single": _compute_packet_error_rate(labels, pred_single, payload_symbols),
            "per_c": _compute_packet_error_rate(labels, pred_cnn, payload_symbols),
            "per_h": _compute_packet_error_rate(labels, pred_hybrid, payload_symbols),
            "util": use_cnn.float().mean().item() * 100.0,
        }

    return stats


def calibrate_global_threshold(
    model,
    simulator,
    calib_dict,
    channel_profile,
    feature_cfg=None,
    eval_batch_size=None,
    hybrid_cfg=None,
    payload_symbols=None,
):
    """calibration dataset을 이용해 global threshold policy를 학습한다."""

    outputs = collect_receiver_outputs(
        model,
        simulator,
        calib_dict,
        channel_profile,
        feature_cfg=feature_cfg,
        eval_batch_size=eval_batch_size,
        hybrid_cfg=hybrid_cfg,
    )
    return calibrate_global_threshold_from_outputs(
        outputs,
        hybrid_cfg=hybrid_cfg,
        payload_symbols=payload_symbols,
    )


def calibrate_confidence_bin_policy(
    model,
    simulator,
    calib_dict,
    channel_profile,
    feature_cfg=None,
    eval_batch_size=None,
    hybrid_cfg=None,
    payload_symbols=None,
):
    """calibration dataset을 이용해 confidence-bin policy를 학습한다."""

    outputs = collect_receiver_outputs(
        model,
        simulator,
        calib_dict,
        channel_profile,
        feature_cfg=feature_cfg,
        eval_batch_size=eval_batch_size,
        hybrid_cfg=hybrid_cfg,
    )
    return calibrate_confidence_bin_policy_from_outputs(
        outputs,
        hybrid_cfg=hybrid_cfg,
        payload_symbols=payload_symbols,
    )


def run_evaluation(
    model,
    simulator,
    test_dict,
    channel_profile,
    policy,
    feature_cfg=None,
    eval_batch_size=None,
    hybrid_cfg=None,
    payload_symbols=None,
):
    """주어진 정책으로 test dataset 전체를 평가해 통계를 반환한다."""

    outputs = collect_receiver_outputs(
        model,
        simulator,
        test_dict,
        channel_profile,
        feature_cfg=feature_cfg,
        eval_batch_size=eval_batch_size,
        hybrid_cfg=hybrid_cfg,
    )
    return summarize_outputs(
        outputs,
        policy,
        hybrid_cfg=hybrid_cfg,
        payload_symbols=payload_symbols,
    )


def benchmark_receivers(
    model,
    simulator,
    reference_dataset,
    channel_profile,
    policy,
    benchmark_cfg=None,
    feature_cfg=None,
    hybrid_cfg=None,
):
    """각 수신기 경로의 상대적 추론 시간을 측정한다.

    여기서 측정하는 시간은 end-to-end 시스템 시간이라기보다,
    동일한 환경에서 각 경로를 비교하기 위한 상대적 기준 시간에 가깝다.
    """

    benchmark_cfg = CFG["benchmark"] if benchmark_cfg is None else benchmark_cfg
    feature_cfg = CFG["feature_bank"] if feature_cfg is None else feature_cfg
    hybrid_cfg = CFG["hybrid"] if hybrid_cfg is None else hybrid_cfg
    conf_type = hybrid_cfg["confidence_type"]

    resolved_profile = simulator.resolve_channel_profile(channel_profile)
    max_cfo_hz = get_max_cfo_hz(simulator, channel_profile)
    cfo_grid, to_grid = simulator.generate_hypothesis_grid(
        max_cfo_hz,
        resolved_profile["max_to_samples"],
        feature_cfg["cfo_steps"],
        feature_cfg["to_steps"],
    )
    helper = simulator.prepare_hypothesis_helper(
        cfo_grid,
        to_grid,
        feature_cfg["patch_size"],
    )

    _, rx_signals = reference_dataset.tensors
    batch_size = min(benchmark_cfg["batch_size"], rx_signals.size(0))
    rx_batch = rx_signals[:batch_size].to(simulator.device)
    compiled_policy = _materialize_policy(policy, simulator.device, torch.float32)

    def single_path():
        """Default LoRa 경로만 실행한다."""

        with torch.inference_mode():
            simulator.baseline_grouped_bin(rx_batch, window_size=feature_cfg["baseline_window"])

    def cnn_path():
        """Full CNN 경로만 실행한다."""

        with torch.inference_mode():
            features = simulator.extract_multi_hypothesis_bank(
                rx_batch,
                helper=helper,
            )
            model(features)

    def hybrid_path():
        """하이브리드 경로를 실제 정책과 동일하게 실행한다."""

        with torch.inference_mode():
            grouped_single, _ = simulator.baseline_grouped_bin(
                rx_batch,
                window_size=feature_cfg["baseline_window"],
            )
            confidence = get_confidence(grouped_single, conf_type=conf_type)
            use_cnn = _policy_mask(confidence, compiled_policy, conf_type=conf_type)
            if torch.any(use_cnn):
                features = simulator.extract_multi_hypothesis_bank(
                    rx_batch[use_cnn],
                    helper=helper,
                )
                model(features)

    return {
        "single_ms": benchmark_callable(
            single_path,
            device=simulator.device,
            warmup=benchmark_cfg["warmup"],
            repeats=benchmark_cfg["repeats"],
        ),
        "cnn_ms": benchmark_callable(
            cnn_path,
            device=simulator.device,
            warmup=benchmark_cfg["warmup"],
            repeats=benchmark_cfg["repeats"],
        ),
        "hybrid_ms": benchmark_callable(
            hybrid_path,
            device=simulator.device,
            warmup=benchmark_cfg["warmup"],
            repeats=benchmark_cfg["repeats"],
        ),
    }
