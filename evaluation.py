import torch
import numpy as np
import torch.nn.functional as F
from config import CFG



def get_confidence(grouped_energy, conf_type='ratio'):
    """
    baseline grouped energy로부터 confidence를 계산하는 함수이다.

    이 함수는 classical receiver가 얼마나 확신을 갖고 있는지를 수치로 표현한다.
    현재는 세 가지 방식을 지원한다.
    1) ratio      : top1 / top2 비율을 사용한다.
    2) norm_margin: (top1 - top2) / top1 형태의 정규화 차이를 사용한다.
    3) entropy    : softmax 분포의 엔트로피를 사용한다.

    연구의 기본값은 ratio이다.
    ratio가 크면 1등 후보가 2등 후보보다 훨씬 크다는 뜻이므로
    classical receiver가 자신 있다고 해석한다.
    """

    # grouped_energy에서 상위 2개의 값만 뽑아낸다.
    top2_values, _ = torch.topk(grouped_energy, 2, dim=1)
    t1, t2 = top2_values[:, 0], top2_values[:, 1]

    if conf_type == 'ratio':
        # 1등 에너지를 2등 에너지로 나눈 값이다.
        # 값이 크면 확신이 높다고 본다.
        return t1 / (t2 + 1e-9)

    if conf_type == 'norm_margin':
        # 1등과 2등의 차이를 1등 값으로 정규화한 것이다.
        # 값이 클수록 1등 후보가 더 뚜렷하다고 본다.
        return (t1 - t2) / (t1 + 1e-9)

    if conf_type == 'entropy':
        # grouped energy를 확률처럼 해석하기 위해 softmax를 취한 뒤,
        # 엔트로피를 계산한다.
        # 엔트로피가 크면 후보들이 고르게 퍼져 있다는 뜻이므로
        # 오히려 불확실성이 높다고 해석한다.
        p = F.softmax(grouped_energy, dim=1)
        return -torch.sum(p * torch.log(p + 1e-9), dim=1)

    # 지정하지 않았을 때는 ratio를 기본값으로 사용한다.
    return t1 / (t2 + 1e-9)



def calibrate_adaptive_policy_joint(model, simulator, calib_dict, max_cfo_hz, conf_type='ratio'):
    """
    calibration 데이터셋을 이용하여 SNR별 adaptive threshold를 자동으로 찾는 함수이다.

    목표는 다음과 같다.
    - SER과 PER이 baseline 대비 지나치게 나빠지지 않도록 제한한다.
    - 그 조건을 만족하는 threshold 중에서 CNN 사용률(utilization)이 가장 낮은 값을 찾는다.

    즉, 단순히 성능만 최대화하는 것이 아니라,
    "성능을 크게 해치지 않으면서 CNN 호출을 최대한 줄이는 정책"을 찾고자 하였다.
    """

    device = simulator.device

    # 평가 모드로 전환한다.
    # dropout, batchnorm 동작을 추론용으로 고정하기 위함이다.
    model.eval()

    # 최종적으로 SNR별 threshold를 저장할 딕셔너리이다.
    policy = {}

    # 중앙 설정값을 불러온다.
    packet_size = CFG["packet_size"]
    eval_batch_size = CFG["eval_batch_size"]

    # calibration 전체에 공통으로 사용할 가설 격자를 만든다.
    cfo_grid, to_grid = simulator.generate_hypothesis_grid(
        max_cfo_hz,
        CFG["max_to_samples"],
        CFG["cfo_steps"],
        CFG["to_steps"],
    )

    # 각 SNR별 데이터셋에 대해 threshold를 따로 찾는다.
    for snr, dataset in calib_dict.items():
        labels, rx_signals = dataset.tensors
        labels, rx_signals = labels.to(device), rx_signals.to(device)

        num_samples = len(labels)
        num_packets = num_samples // packet_size

        # 미니배치로 나누어 계산한 결과를 모을 리스트이다.
        pred_g_list, pred_c_list, conf_list = [], [], []

        print(f" -> SNR {snr:3d}dB에서 최적 임계값을 탐색 중.")

        with torch.no_grad():
            # 한 번에 전체를 처리하면 메모리 사용량이 커지므로
            # 평가도 배치 분할 방식으로 진행하였다.
            for i in range(0, num_samples, eval_batch_size):
                end = min(i + eval_batch_size, num_samples)
                rx_batch = rx_signals[i:end]

                # 1. classical grouped-bin 복조 결과를 계산한다.
                grouped_energy, _ = simulator.baseline_grouped_bin(rx_batch)
                pred_g_list.append(torch.argmax(grouped_energy, dim=1))
                conf_list.append(get_confidence(grouped_energy, conf_type))

                # 2. 다중 가설 특징맵을 만들고 CNN 예측을 구한다.
                features = simulator.extract_multi_hypothesis_bank(
                    rx_batch,
                    cfo_grid,
                    to_grid,
                    CFG["patch_size"],
                )
                pred_c_list.append(torch.argmax(model(features), dim=1))

            # 배치별 결과를 하나의 텐서로 합친다.
            pred_g = torch.cat(pred_g_list)
            pred_c = torch.cat(pred_c_list)
            conf = torch.cat(conf_list)

            # ------------------------------------------------------------
            # baseline 성능 계산
            # ------------------------------------------------------------
            ser_g = 1.0 - (pred_g == labels).float().mean().item()
            ser_c = 1.0 - (pred_c == labels).float().mean().item()

            labels_pkt = labels.view(num_packets, packet_size)
            per_g = (torch.any(labels_pkt != pred_g.view(num_packets, packet_size), dim=1)).float().mean().item()
            per_c = (torch.any(labels_pkt != pred_c.view(num_packets, packet_size), dim=1)).float().mean().item()

            # baseline과 CNN 중 더 좋은 쪽을 기준 성능으로 잡는다.
            base_ser = min(ser_g, ser_c)
            base_per = min(per_g, per_c)

            # ------------------------------------------------------------
            # 허용 오차 설정
            # ------------------------------------------------------------
            # 극저 SNR에서는 base error 자체가 매우 크므로
            # 상대 10% 허용만 쓰면 너무 느슨해질 수 있다.
            # 그래서 -21 dB 같은 구간은 절대 마진을 사용하였다.
            if snr <= -20:
                target_ser = base_ser + 0.005
                target_per = base_per + 0.01
            else:
                # 그 외 구간에서는 base 성능 대비 10% 열화 허용과
                # 최소 절대 마진을 동시에 고려하였다.
                target_ser = max(base_ser * 1.10, base_ser + 0.0005)
                target_per = max(base_per * 1.10, base_per + 0.005)

            # ------------------------------------------------------------
            # threshold 후보 탐색 준비
            # ------------------------------------------------------------
            # best_th는 조건을 만족하는 후보 중 util이 최소인 값을 저장한다.
            best_th = 3.0 if conf_type == 'ratio' else 1.0
            min_util = 100.0

            # fallback_th는 모든 조건을 만족하는 후보가 없을 때 사용할 후보이다.
            # penalty가 가장 작은 threshold를 저장한다.
            fallback_th = best_th
            min_penalty = float('inf')
            valid_found = False

            # ratio는 보통 1.0 ~ 3.0 범위를 쓰고,
            # 다른 지표는 0.0 ~ 1.0 범위를 탐색한다.
            search_space = torch.linspace(1.0, 3.0, 41) if conf_type == 'ratio' else torch.linspace(0.0, 1.0, 41)

            # ------------------------------------------------------------
            # threshold sweep
            # ------------------------------------------------------------
            for th in search_space:
                # ratio와 norm_margin은 값이 작을수록 불확실하다고 보고 CNN을 사용한다.
                # entropy는 값이 클수록 불확실하므로 방향이 반대이다.
                use_cnn = (conf < th) if conf_type != 'entropy' else (conf > th)

                # threshold 조건에 따라 baseline과 CNN 결과 중 하나를 선택한다.
                pred_h = torch.where(use_cnn, pred_c, pred_g)

                # hybrid SER / PER / utilization을 계산한다.
                s_h = 1.0 - (pred_h == labels).float().mean().item()
                p_h = (torch.any(labels_pkt != pred_h.view(num_packets, packet_size), dim=1)).float().mean().item()
                util = use_cnn.float().mean().item() * 100

                # 조건을 만족하면서 util이 가장 작은 후보를 정답으로 채택한다.
                if s_h <= target_ser and p_h <= target_per and util < min_util:
                    best_th = th.item()
                    min_util = util
                    valid_found = True

                # 조건을 만족하는 후보가 하나도 없을 수도 있으므로,
                # 그 경우를 대비해 penalty가 가장 작은 후보도 따로 추적한다.
                penalty = max(0, s_h - target_ser) * 10 + max(0, p_h - target_per) * 5 + (util / 100)
                if penalty < min_penalty:
                    min_penalty = penalty
                    fallback_th = th.item()

            # 최종 threshold를 정책 딕셔너리에 저장한다.
            # 유효 후보가 있으면 best_th를 쓰고,
            # 없으면 fallback_th를 쓴다.
            policy[snr] = round(best_th if valid_found else fallback_th, 2)

    return policy



def run_evaluation(model, simulator, test_dict, max_cfo_hz, policy, conf_type='ratio'):
    """
    주어진 policy로 최종 평가를 수행하는 함수이다.

    입력 policy는 두 가지 형태를 받을 수 있다.
    1) float  : 모든 SNR에 같은 고정 threshold를 적용한다.
    2) dict   : SNR별 adaptive threshold를 적용한다.

    반환값은 SNR별 통계 정보를 담은 딕셔너리이다.
    여기에는 SER, PER, utilization, threshold가 모두 들어간다.
    """

    device = simulator.device
    model.eval()
    stats = {}

    packet_size = CFG["packet_size"]
    eval_batch_size = CFG["eval_batch_size"]

    # 평가에 사용할 다중 가설 격자를 만든다.
    cfo_grid, to_grid = simulator.generate_hypothesis_grid(
        max_cfo_hz,
        CFG["max_to_samples"],
        CFG["cfo_steps"],
        CFG["to_steps"],
    )

    for snr, dataset in test_dict.items():
        labels, rx_signals = dataset.tensors
        labels, rx_signals = labels.to(device), rx_signals.to(device)

        num_samples = len(labels)
        num_packets = num_samples // packet_size

        # policy가 dict이면 현재 SNR에 해당하는 threshold를 사용한다.
        # 아니면 같은 고정 threshold를 모든 SNR에 사용한다.
        th = policy.get(snr, 1.5) if isinstance(policy, dict) else policy

        pred_g_list, pred_c_list, conf_list = [], [], []

        with torch.no_grad():
            for i in range(0, num_samples, eval_batch_size):
                end = min(i + eval_batch_size, num_samples)
                rx_batch = rx_signals[i:end]

                # classical grouped-bin 복조
                grouped_energy, _ = simulator.baseline_grouped_bin(rx_batch)
                pred_g_list.append(torch.argmax(grouped_energy, dim=1))
                conf_list.append(get_confidence(grouped_energy, conf_type))

                # CNN 복조
                features = simulator.extract_multi_hypothesis_bank(
                    rx_batch,
                    cfo_grid,
                    to_grid,
                    CFG["patch_size"],
                )
                pred_c_list.append(torch.argmax(model(features), dim=1))

            pred_g = torch.cat(pred_g_list)
            pred_c = torch.cat(pred_c_list)
            conf = torch.cat(conf_list)

            # confidence와 threshold를 비교하여 CNN 사용 여부를 정한다.
            use_cnn = (conf < th) if conf_type != 'entropy' else (conf > th)

            # hybrid 결과를 만든다.
            pred_h = torch.where(use_cnn, pred_c, pred_g)

            # packet 단위 비교를 위해 packet_size 기준으로 reshape한다.
            labels_pkt = labels.view(num_packets, packet_size)

            stats[snr] = {
                # baseline SER
                "ser_g": 1.0 - (pred_g == labels).float().mean().item(),

                # full CNN SER
                "ser_c": 1.0 - (pred_c == labels).float().mean().item(),

                # hybrid SER
                "ser_h": 1.0 - (pred_h == labels).float().mean().item(),

                # baseline PER
                "per_g": (torch.any(labels_pkt != pred_g.view(num_packets, packet_size), dim=1)).float().mean().item(),

                # full CNN PER
                "per_c": (torch.any(labels_pkt != pred_c.view(num_packets, packet_size), dim=1)).float().mean().item(),

                # hybrid PER
                "per_h": (torch.any(labels_pkt != pred_h.view(num_packets, packet_size), dim=1)).float().mean().item(),

                # 현재 SNR에서 CNN이 실제로 사용된 비율
                "util": use_cnn.float().mean().item() * 100,

                # 사용된 threshold 값
                "th": th,
            }

    return stats
