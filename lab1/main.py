"""
Лабораторна робота №1. Інтерполяція кубічними сплайнами.

Профіль висоти маршруту від станції Заросляк до вершини гори Говерла:
- отримання висот через Open-Elevation API;
- побудова природного кубічного сплайна (коефіцієнти c_i — методом прогонки);
- дослідження впливу кількості вузлів (10, 15, 20) на точність;
- характеристики маршруту, аналіз градієнта, механічна робота підйому.
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "elevation_data.json")
TAB_FILE = os.path.join(BASE_DIR, "tabulation.txt")
IMG_DIR = os.path.join(BASE_DIR, "images")

URL = ("https://api.open-elevation.com/api/v1/lookup?locations="
       "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|"
       "48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|"
       "48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
       "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|"
       "48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|"
       "48.160250,24.500106")


# ---------------------------------------------------------------------------
# 1–3. Отримання даних і табуляція
# ---------------------------------------------------------------------------

def fetch_elevations():
    """Запит до Open-Elevation API; якщо API недоступне — читаємо локальну копію."""
    try:
        response = requests.get(URL, timeout=60)
        response.raise_for_status()
        data = response.json()
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print("Дані отримано з API")
    except Exception as e:
        print(f"API недоступне ({e}), використовую {os.path.basename(CACHE_FILE)}")
        with open(CACHE_FILE, encoding="utf-8") as f:
            data = json.load(f)
    return data["results"]


def haversine(lat1, lon1, lat2, lon2):
    """Відстань між двома точками на сфері (м)."""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ---------------------------------------------------------------------------
# 6–9. Кубічний сплайн
# ---------------------------------------------------------------------------

def build_system(x, y):
    """Коефіцієнти тридіагональної системи для c_1..c_{n-1} (c_0 = c_n = 0)."""
    h = np.diff(x)
    m = len(x) - 2                          # кількість невідомих
    alpha = np.zeros(m)                     # піддіагональ (alpha[0] = 0)
    beta = np.zeros(m)                      # головна діагональ
    gamma = np.zeros(m)                     # наддіагональ (gamma[-1] = 0)
    delta = np.zeros(m)                     # вільні члени
    for k in range(m):
        i = k + 1                           # номер внутрішнього вузла
        alpha[k] = h[i - 1] if k > 0 else 0.0
        beta[k] = 2 * (h[i - 1] + h[i])
        gamma[k] = h[i] if k < m - 1 else 0.0
        delta[k] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    return h, alpha, beta, gamma, delta


def thomas(alpha, beta, gamma, delta, verbose=False):
    """Метод прогонки для тридіагональної системи."""
    m = len(beta)
    A = np.zeros(m)
    B = np.zeros(m)
    # пряма прогонка
    A[0] = -gamma[0] / beta[0]
    B[0] = delta[0] / beta[0]
    for i in range(1, m):
        denom = alpha[i] * A[i - 1] + beta[i]
        A[i] = -gamma[i] / denom
        B[i] = (delta[i] - alpha[i] * B[i - 1]) / denom
    # зворотна прогонка
    x = np.zeros(m)
    x[-1] = B[-1]
    for i in range(m - 2, -1, -1):
        x[i] = A[i] * x[i + 1] + B[i]

    if verbose:
        print("\nПрогонкові коефіцієнти:")
        print(" i |      A_i      |      B_i")
        for i in range(m):
            print(f"{i + 1:2d} | {A[i]:13.8f} | {B[i]:13.8f}")
    return x


def cubic_spline(x, y, verbose=False):
    """Коефіцієнти a, b, c, d природного кубічного сплайна."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    h, alpha, beta, gamma, delta = build_system(x, y)

    c_full = np.zeros(len(x))
    c_full[1:-1] = thomas(alpha, beta, gamma, delta)

    a = y[:-1].copy()
    c = c_full[:-1].copy()
    b = (y[1:] - y[:-1]) / h - h * (c_full[1:] + 2 * c_full[:-1]) / 3
    d = (c_full[1:] - c_full[:-1]) / (3 * h)

    if verbose:
        print(" i |   [x_{i-1}, x_i], м   |     a_i     |      b_i      |      c_i       |      d_i")
        for i in range(len(h)):
            print(f"{i + 1:2d} | [{x[i]:7.1f}, {x[i + 1]:7.1f}] | {a[i]:11.4f} | {b[i]:13.8f} | "
                  f"{c[i]:14.10f} | {d[i]:15.12f}")
    return a, b, c, d


def spline_eval(x_nodes, coeffs, xx):
    """Значення сплайна в точках xx."""
    a, b, c, d = coeffs
    idx = np.clip(np.searchsorted(x_nodes, xx, side="right") - 1, 0, len(a) - 1)
    t = xx - x_nodes[idx]
    return a[idx] + b[idx] * t + c[idx] * t ** 2 + d[idx] * t ** 3


def save_fig(fig, name):
    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(os.path.join(IMG_DIR, name), dpi=120, bbox_inches="tight")


def main():
    plt.rcParams["axes.grid"] = True

    # ---- 1–3. Дані та табуляція ----
    results = fetch_elevations()
    n = len(results)
    print("Кількість вузлів:", n)

    lines = ["№ | Latitude | Longitude | Elevation (m)"]
    for i, p in enumerate(results):
        lines.append(f"{i:2d} | {p['latitude']:.6f} | {p['longitude']:.6f} | {p['elevation']:.2f}")
    print("\nТабуляція вузлів:")
    print("\n".join(lines))

    # ---- 4. Кумулятивна відстань ----
    coords = [(p["latitude"], p["longitude"]) for p in results]
    elevations = [p["elevation"] for p in results]
    distances = [0.0]
    for i in range(1, n):
        distances.append(distances[-1] + haversine(*coords[i - 1], *coords[i]))

    lines2 = ["№ | Distance (m) | Elevation (m)"]
    for i in range(n):
        lines2.append(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")
    print("\nТабуляція (відстань, висота):")
    print("\n".join(lines2))

    with open(TAB_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n" + "\n".join(lines2) + "\n")
    print(f"\nТабуляцію записано у файл {os.path.basename(TAB_FILE)}")

    x_all = np.array(distances)
    y_all = np.array(elevations)

    # ---- 5. Графік дискретного профілю ----
    fig = plt.figure(figsize=(11, 5))
    plt.plot(x_all, y_all, "o-")
    plt.xlabel("Кумулятивна відстань, м")
    plt.ylabel("Висота, м")
    plt.title("Профіль висоти: Заросляк → Говерла (вузли)")
    save_fig(fig, "profile_nodes.png")

    # ---- 6. Коефіцієнти тридіагональної системи ----
    h, alpha, beta, gamma, delta = build_system(x_all, y_all)
    print("\nКоефіцієнти тридіагональної системи:")
    print(" k |    alpha    |    beta     |    gamma    |    delta")
    for k in range(len(beta)):
        print(f"{k + 1:2d} | {alpha[k]:11.4f} | {beta[k]:11.4f} | {gamma[k]:11.4f} | {delta[k]:11.6f}")

    # ---- 7. Метод прогонки ----
    stable = np.all(np.abs(beta) >= np.abs(alpha) + np.abs(gamma))
    print("\nУмова діагональної переваги виконана:", stable)
    c_inner = thomas(alpha, beta, gamma, delta, verbose=True)
    print("\nКоефіцієнти c_i (внутрішні вузли):")
    for i, ci in enumerate(c_inner, start=2):
        print(f"c_{i} = {ci:.10f}")

    M = np.diag(beta) + np.diag(alpha[1:], -1) + np.diag(gamma[:-1], 1)
    print("Максимальна нев'язка |Mc - delta|:", np.max(np.abs(M @ c_inner - delta)))

    # ---- 8–9. Коефіцієнти сплайнів ----
    print("\nКоефіцієнти кубічних сплайнів (усі вузли):")
    coeffs_full = cubic_spline(x_all, y_all, verbose=True)
    print("Макс. відхилення сплайна у вузлах:",
          np.max(np.abs(spline_eval(x_all, coeffs_full, x_all) - y_all)))

    # ---- 10. Сплайни з 10, 15, 20 вузлами ----
    xx = np.linspace(x_all[0], x_all[-1], 2000)
    node_counts = [10, 15, 20]
    splines = {}
    for k in node_counts:
        idx = np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))
        xk, yk = x_all[idx], y_all[idx]
        print(f"\n===== {k} вузлів =====")
        splines[k] = (xk, yk, cubic_spline(xk, yk, verbose=True))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for ax, k in zip(axes, node_counts):
        xk, yk, cf = splines[k]
        ax.plot(xx, spline_eval(xk, cf, xx), label="Кубічний сплайн")
        ax.plot(x_all, y_all, "o", ms=4, color="gray", alpha=0.6, label=f"Усі дані ({n})")
        ax.plot(xk, yk, "o", color="tab:red", label=f"Вузли ({k})")
        ax.set_title(f"{k} вузлів")
        ax.set_xlabel("Відстань, м")
        ax.legend()
    axes[0].set_ylabel("Висота, м")
    fig.tight_layout()
    save_fig(fig, "splines_10_15_20.png")

    # ---- 11–12. Вплив кількості вузлів на точність ----
    f_ref = spline_eval(x_all, coeffs_full, xx)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(xx, f_ref, "k", lw=2.2, label=f"f(x) — сплайн за {n} вузлами")
    print("\n Вузлів | max ε(x), м | середня ε(x), м | max похибка у пропущених вузлах, м")
    for k in node_counts:
        xk, yk, cf = splines[k]
        phi = spline_eval(xk, cf, xx)
        err = np.abs(f_ref - phi)
        axes[0].plot(xx, phi, "--", label=f"φ(x), {k} вузлів")
        axes[1].plot(xx, err, label=f"{k} вузлів")
        missing = ~np.isin(x_all, xk)
        node_err = np.max(np.abs(spline_eval(xk, cf, x_all[missing]) - y_all[missing])) if missing.any() else 0.0
        print(f"   {k:3d}  | {err.max():11.3f} | {err.mean():15.3f} | {node_err:10.3f}")
    axes[0].set_ylabel("Висота, м")
    axes[0].set_title("Функція та її наближення")
    axes[0].legend()
    axes[1].set_ylabel("ε(x) = |f(x) − φ(x)|, м")
    axes[1].set_xlabel("Відстань, м")
    axes[1].set_title("Похибка інтерполяції")
    axes[1].legend()
    fig.tight_layout()
    save_fig(fig, "error.png")

    # ---- Додатково 1. Характеристики маршруту ----
    print("\nЗагальна довжина маршруту (м):", round(distances[-1], 2))
    total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, n))
    total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, n))
    print("Сумарний набір висоти (м):", round(total_ascent, 2))
    print("Сумарний спуск (м):", round(total_descent, 2))

    # ---- Додатково 2. Аналіз градієнта ----
    yy_full = f_ref
    grad_full = np.gradient(yy_full, xx) * 100
    print("\nМаксимальний підйом (%):", round(np.max(grad_full), 2))
    print("Максимальний спуск (%):", round(np.min(grad_full), 2))
    print("Середній градієнт (%):", round(np.mean(np.abs(grad_full)), 2))

    steep = np.abs(grad_full) > 15
    edges = np.flatnonzero(np.diff(np.r_[0, steep.astype(int), 0]))
    print("Ділянки з крутизною > 15%:")
    for s, e in zip(edges[::2], edges[1::2]):
        seg = grad_full[s:e]
        print(f"  {xx[s]:7.1f} – {xx[e - 1]:7.1f} м (довжина {xx[e - 1] - xx[s]:6.1f} м), "
              f"макс. {seg[np.argmax(np.abs(seg))]:.1f}%")
    print(f"Частка маршруту з крутизною > 15%: {steep.mean() * 100:.1f}%")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(xx, yy_full)
    axes[0].fill_between(xx, yy_full.min(), yy_full, where=steep, color="tab:red", alpha=0.25, label="> 15%")
    axes[0].plot(x_all, y_all, "o", color="tab:blue", ms=4)
    axes[0].set_ylabel("Висота, м")
    axes[0].set_title("Гладкий профіль висоти маршруту")
    axes[0].legend()
    axes[1].plot(xx, grad_full, color="tab:green")
    axes[1].axhline(15, color="tab:red", ls="--")
    axes[1].axhline(-15, color="tab:red", ls="--")
    axes[1].set_ylabel("Градієнт, %")
    axes[1].set_xlabel("Відстань, м")
    fig.tight_layout()
    save_fig(fig, "gradient.png")

    # ---- Додатково 3. Механічна енергія підйому ----
    mass = 80
    g = 9.81
    energy = mass * g * total_ascent
    print("\nМеханічна робота (Дж):", round(energy, 1))
    print("Механічна робота (кДж):", round(energy / 1000, 2))
    print("Енергія (ккал):", round(energy / 4184, 2))

    plt.show()


if __name__ == "__main__":
    main()
