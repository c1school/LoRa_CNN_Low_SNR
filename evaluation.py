import matplotlib.pyplot as plt
import torch


def calculate_snr_for_target_ser(snrs, sers, target_ser):
    """
    특정 SER(예: 1e-1, 1e-2)를 달성하는 데 필요한 SNR을
    선형 보간으로 계산하는 함수.

    baseline 대비 CNN이 몇 dB 개선되었는가를 수치화하기 위한 도구임.
    """
    for i in range(len(sers) - 1):
        if sers[i] >= target_ser >= sers[i + 1]:
            slope = (snrs[i + 1] - snrs[i]) / (sers[i + 1] - sers[i] + 1e-12)
            return snrs[i] + slope * (target_ser - sers[i])
    return None


def evaluate_ablation_model(model_complex, model_mag, simulator, snr_list, impairment_config, benchmark_name):
    """
    같은 noisy sample에 대해 아래 네 방법을 동시에 비교함.

    1) Naive FFT baseline
    2) Grouped-bin classical baseline
    3) Magnitude-only CNN
    4) Complex-input CNN

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_complex.to(device).eval()
    model_mag.to(device).eval()

    # 각 방법별 SER 곡선 저장용
    results = {"Complex CNN": [], "Mag CNN": [], "Grouped": [], "Naive": []}

    print(f"\n[정밀 평가 시작 : {benchmark_name}]")
    print("-" * 80)

    for snr in snr_list:
        # 저오류율 구간일수록 더 많은 샘플을 써야 SER가 통계적으로 의미 있음
        if snr >= -10:
            num_test_samples = 50000
        elif snr >= -15:
            num_test_samples = 10000
        else:
            num_test_samples = 3000

        print(f"SNR {snr:3d} dB 평가 중... (Monte Carlo: {num_test_samples})")

        # 정답 개수 카운터
        cor = {"Complex CNN": 0, "Mag CNN": 0, "Grouped": 0, "Naive": 0}

        batch_size = 1000
        num_batches = num_test_samples // batch_size

        with torch.no_grad():
            for _ in range(num_batches):
                labels_np = torch.randint(0, simulator.M, (batch_size,), dtype=torch.long).numpy()
                features_c, features_m = [], []

                for lbl in labels_np:
                    clean_sig = simulator.generate_symbol(int(lbl))
                    noisy_sig = simulator.apply_impaired_channel(clean_sig, snr, impairment_config)

                    # baseline 평가
                    if simulator.baseline_demod_naive(noisy_sig) == int(lbl):
                        cor["Naive"] += 1
                    if simulator.baseline_demod_grouped_bin(noisy_sig, window_size=2) == int(lbl):
                        cor["Grouped"] += 1

                    # CNN 입력 특징 추출
                    features_c.append(torch.tensor(simulator.dechirp_and_fft_complex(noisy_sig), dtype=torch.float32))
                    features_m.append(torch.tensor(simulator.dechirp_and_fft_mag(noisy_sig), dtype=torch.float32))

                # CNN 평가
                out_c = model_complex(torch.stack(features_c).to(device))
                out_m = model_mag(torch.stack(features_m).to(device))
                lbl_t = torch.tensor(labels_np, dtype=torch.long).to(device)

                cor["Complex CNN"] += (torch.max(out_c.data, 1)[1] == lbl_t).sum().item()
                cor["Mag CNN"] += (torch.max(out_m.data, 1)[1] == lbl_t).sum().item()

        # SER 계산
        for key in results.keys():
            results[key].append(1.0 - (cor[key] / num_test_samples))

        print(
            f"  -> SER [Comp CNN]:{results['Complex CNN'][-1]:.5f} | "
            f"[Mag CNN]:{results['Mag CNN'][-1]:.5f} | [Grp]:{results['Grouped'][-1]:.5f}"
        )

    # 그래프 저장
    filename = f"ser_curve_{benchmark_name.lower().replace(' ', '_')}.png"
    plt.figure(figsize=(10, 7))
    plt.semilogy(snr_list, results["Naive"], marker="x", linestyle=":", color="gray", alpha=0.7, label="Naive FFT")
    plt.semilogy(snr_list, results["Complex CNN"], marker="o", linestyle="-", color="red", linewidth=2, label="Complex Input CNN")

    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel("Signal-to-Noise Ratio (SNR) [dB]", fontsize=12)
    plt.ylabel("Symbol Error Rate (SER)", fontsize=12)
    plt.title(f"LoRa Demodulation Performance: {benchmark_name}", fontsize=14)
    plt.ylim([5e-5, 1.1])
    plt.legend(fontsize=11, loc="lower left")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    # Threshold Gain 정량화 표 출력
    print(f"\n[{benchmark_name}] Threshold Gain 분석표")
    print("=" * 60)
    print(f"{'Method':<20} | {'SNR for SER=1e-1':<15} | {'SNR for SER=1e-2':<15}")
    print("-" * 60)
    for key in ["Grouped", "Mag CNN", "Complex CNN"]:
        snr_1e1 = calculate_snr_for_target_ser(snrs=snr_list, sers=results[key], target_ser=1e-1)
        snr_1e2 = calculate_snr_for_target_ser(snrs=snr_list, sers=results[key], target_ser=1e-2)

        str_1e1 = f"{snr_1e1:.2f} dB" if snr_1e1 is not None else "N/A"
        str_1e2 = f"{snr_1e2:.2f} dB" if snr_1e2 is not None else "N/A"

        print(f"{key:<20} | {str_1e1:<15} | {str_1e2:<15}")
    print("=" * 60)

    return results
