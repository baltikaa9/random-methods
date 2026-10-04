import math

import matplotlib.pyplot as plt
import numpy as np

from .lemer import LemerGenerator


def generate_uniform(a: int, b: int, n: int, gen: LemerGenerator) -> list[float]:
    return [a + (b - a) * gen.next_random() for _ in range(n)]


def generate_exp(lam: int, n: int, gen: LemerGenerator) -> list[float]:
    return [-(1 / lam) * math.log(max(gen.next_random(), 1e-10)) for _ in range(n)]


def generate_normal(mu: int, sigma: int, n: int, gen: LemerGenerator) -> list[float]:
    return [
        mu + sigma * (sum(gen.next_random() for _ in range(12)) - 6) for _ in range(n)
    ]


def mean(data: list[float]) -> float:
    return sum(data) / len(data)


def variance(data: list[float]) -> float:
    if (n := len(data)) <= 1:
        return 0.0

    m = mean(data)
    return sum((x - m) * (x - m) for x in data) / (n - 1)


def main() -> None:
    seed = 42
    a, b = 4, 8
    lam = 5
    mu, sigma = 10, 10

    mean_theory_uni = (a + b) / 2
    var_theory_uni = (b - a) ** 2 / 12

    mean_theory_exp = 1 / lam
    var_theory_exp = 1 / (lam**2)

    mean_theory_norm = mu
    var_theory_norm = sigma**2

    sample_sizes = [10, 20, 50, 100, 1_000, 10_000, 100_000]

    means_uni, vars_uni = [], []
    means_exp, vars_exp = [], []
    means_norm, vars_norm = [], []

    for n in sample_sizes:
        uni = generate_uniform(a, b, n, LemerGenerator(seed))
        exp = generate_exp(lam, n, LemerGenerator(seed))
        norm = generate_normal(mu, sigma, n, LemerGenerator(seed))

        means_uni.append(mean(uni))
        vars_uni.append(variance(uni))

        means_exp.append(mean(exp))
        vars_exp.append(variance(exp))

        means_norm.append(mean(norm))
        vars_norm.append(variance(norm))

        print(
            f"N={n:6d} | Равн: Mx={means_uni[-1]:.4f}, Dx={vars_uni[-1]:.4f} | "
            f"Эксп: Mx={means_exp[-1]:.4f}, Dx={vars_exp[-1]:.4f} | "
            f"Норм: Mx={means_norm[-1]:.4f}, Dx={vars_norm[-1]:.4f}"
        )

    def relative_error(estimated, theoretical):
        return abs(estimated - theoretical) / abs(theoretical) * 100

    err_m_uni = relative_error(means_uni[-1], mean_theory_uni)
    err_v_uni = relative_error(vars_uni[-1], var_theory_uni)
    print(f"Равномерное: ε(Mx) = {err_m_uni:.2f}%, ε(Dx) = {err_v_uni:.2f}%")

    err_m_exp = relative_error(means_exp[-1], mean_theory_exp)
    err_v_exp = relative_error(vars_exp[-1], var_theory_exp)
    print(f"Экспоненциальное: ε(Mx) = {err_m_exp:.2f}%, ε(Dx) = {err_v_exp:.2f}%")

    err_m_norm = relative_error(means_norm[-1], mean_theory_norm)
    err_v_norm = relative_error(vars_norm[-1], var_theory_norm)
    print(f"Нормальное: ε(Mx) = {err_m_norm:.2f}%, ε(Dx) = {err_v_norm:.2f}%")

    _, axes1 = plt.subplots(2, 3, figsize=(15, 10))

    plot_data = [
        (
            axes1[0, 0],
            sample_sizes,
            means_uni,
            mean_theory_uni,
            f"Mx теор. = {mean_theory_uni}",
            "Равномерное: Mx от N",
            "o",
        ),
        (
            axes1[0, 1],
            sample_sizes,
            means_exp,
            mean_theory_exp,
            f"Mx теор. = {mean_theory_exp}",
            "Экспоненциальное: Mx от N",
            "s",
        ),
        (
            axes1[0, 2],
            sample_sizes,
            means_norm,
            mean_theory_norm,
            f"Mx теор. = {mean_theory_norm}",
            "Нормальное: Mx от N",
            "^",
        ),
        (
            axes1[1, 0],
            sample_sizes,
            vars_uni,
            var_theory_uni,
            f"Dx теор. = {var_theory_uni:.4f}",
            "Равномерное: Dx от N",
            "o",
        ),
        (
            axes1[1, 1],
            sample_sizes,
            vars_exp,
            var_theory_exp,
            f"Dx теор. = {var_theory_exp:.4f}",
            "Экспоненциальное: Dx от N",
            "s",
        ),
        (
            axes1[1, 2],
            sample_sizes,
            vars_norm,
            var_theory_norm,
            f"Dx теор. = {var_theory_norm}",
            "Нормальное: Dx от N",
            "^",
        ),
    ]

    for ax, xs, ys, theor, theor_label, title, marker in plot_data:
        ax.semilogx(xs, ys, f"{marker}-", label="Выборочное")
        ax.axhline(y=theor, color="r", linestyle="--", label=theor_label)
        ax.set_xscale("log")  # ← логарифмическая шкала
        ax.set_title(title)
        ax.set_xlabel("N (log scale)")
        ax.set_ylabel("Значение")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("lab1_graphs.png", dpi=150)

    n = 1_000

    uni_final = generate_uniform(a, b, n, LemerGenerator(seed))
    exp_final = generate_exp(lam, n, LemerGenerator(seed))
    norm_final = generate_normal(mu, sigma, n, LemerGenerator(seed))

    K = round(1 + 3.2 * math.log10(n))

    _, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    axes2[0].hist(
        uni_final,
        bins=K,
        density=True,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        label="Плотность распределения",
    )
    x_uni = np.linspace(a - 1, b + 1, 100)
    y_uni = [1 / (b - a) if a <= x <= b else 0 for x in x_uni]
    axes2[0].plot(x_uni, y_uni, "r-", linewidth=2, label="Теор. плотность")
    axes2[0].set_title(f"Равномерное ({a}, {b})")
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.5)

    axes2[1].hist(
        exp_final,
        bins=K,
        density=True,
        alpha=0.7,
        color="lightgreen",
        edgecolor="black",
        label="Плотность распределения",
    )
    x_exp = np.linspace(0, max(exp_final) * 1.1, 100)
    y_exp = [lam * math.exp(-lam * x) for x in x_exp]
    axes2[1].plot(x_exp, y_exp, "r-", linewidth=2, label="Теор. плотность")
    axes2[1].set_title(f"Экспоненциальное (λ={lam})")
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.5)

    axes2[2].hist(
        norm_final,
        bins=K,
        density=True,
        alpha=0.7,
        color="salmon",
        edgecolor="black",
        label="Плотность распределения",
    )
    x_norm = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
    y_norm = [
        1
        / (sigma * math.sqrt(2 * math.pi))
        * math.exp(-((x - mu) ** 2) / (2 * sigma**2))
        for x in x_norm
    ]
    axes2[2].plot(x_norm, y_norm, "r-", linewidth=2, label="Теор. плотность")
    axes2[2].set_title(f"Нормальное (μ={mu}, σ={sigma})")
    axes2[2].legend()
    axes2[2].grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig("lab1_histograms.png", dpi=150)

    plt.show()


if __name__ == "__main__":
    main()
