import matplotlib.pyplot as plt
import random
import time


# A
def dijkstra(V, E, start):
    V=set(V)
    d = {node: float('inf') for node in V}
    pi = {node: None for node in V}
    S = set()
    d[start] = 0
    while len(S) < len(V):
        unvisited = V - S
        u = min(unvisited, key=lambda node: d[node])
        if d[u] == float('inf'):
            break
        S.add(u)
        for v, w in E[u]:
            if d[u] + w < d[v]:
                d[v] = d[u] + w
                pi[v] = u
    return d, pi


def make_graph(n, p, seed=0):
    random.seed(seed)
    V = list(range(n))
    E = {u: [] for u in V}
    m = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if random.random() < p:
                w = random.randint(1, 10)
                E[i].append((j, float(w)))
                m += 1
    return V, E, m

def time_trial(n, p, trials=3):
    ts,ms = [],[]
    for t in range(trials):
        V, E, m = make_graph(n, p, seed=100+t)
        ms.append(m)
        t0 = time.perf_counter()
        dijkstra(V, E, start=0)
        t1 = time.perf_counter()
        ts.append(t1 - t0)
        mean_t = sum(ts)/len(ts)
        m_avg = sum(ms)/len(ms)
    return mean_t, m_avg

def experiment():
    sizes = [80, 120, 160, 200, 240, 280, 320]
    ps = [0.05, 0.1, 0.2, 0.4]
    rows = []
    for p in ps:
        for n in sizes:
            mean_t, m_avg = time_trial(n, p)   # if your time_one returns only time, set m_avg = 0
            rows.append({
                "n": n,
                "p": p,
                "m": int(m_avg),
                "n2": n*n,
                "n2_plus_m": n*n + int(m_avg),
                "mean_time": mean_t
            })
    return rows

def plot_time_vs_n(rows):
    plt.figure()
    # group by p
    ps = sorted({r["p"] for r in rows})
    for p in ps:
        sub = sorted([r for r in rows if r["p"] == p], key=lambda r: r["n"])
        xs = [r["n"] for r in sub]
        ys = [r["mean_time"] for r in sub]
        plt.plot(xs, ys, marker="o", label=f"p={p}")
    plt.xlabel("|V| (n)")
    plt.ylabel("Mean time (s)")
    plt.title("Dijkstra (array min): Time vs |V|")
    plt.legend()
    plt.tight_layout()
    plt.savefig("time vs n.png")
    plt.close()

def plot_time_vs_n2(rows):
    plt.figure()
    ps = sorted({r["p"] for r in rows})
    for p in ps:
        sub = sorted([r for r in rows if r["p"] == p], key=lambda r: r["n2"])
        xs = [r["n2"] for r in sub]
        ys = [r["mean_time"] for r in sub]
        plt.plot(xs, ys, marker="o", label=f"p={p}")
    plt.xlabel("|V|^2")
    plt.ylabel("Mean time (s)")
    plt.title("Time vs |V|^2 (should look ~linear)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("time vs n^2.png")
    plt.close()

def plot_time_vs_n2_plus_m(rows):
    plt.figure()
    ps = sorted({r["p"] for r in rows})
    for p in ps:
        sub = sorted([r for r in rows if r["p"] == p], key=lambda r: r["n2_plus_m"])
        xs = [r["n2_plus_m"] for r in sub]
        ys = [r["mean_time"] for r in sub]
        plt.plot(xs, ys, marker="o", label=f"p={p}")
    plt.xlabel("|V|^2 + |E| (≈ n^2 + m)")
    plt.ylabel("Mean time (s)")
    plt.title("Time vs (|V|^2 + |E|)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("time vs n^2 + m.png")
    plt.close()

#C
def time_comparision(n, p, trials=3):
    times_array, times_heap, ms = [], [], []
    for t in range(trials):
        V, E, m = make_graph(n, p, seed=100+t)
        ms.append(m)

        # Array
        t0 = time.perf_counter()
        dijkstra(V, E, start=0)
        t1 = time.perf_counter()
        times_array.append(t1 - t0)

        # Heap
        t2 = time.perf_counter()
        #dijkstra_heap(V, E, start=0)
        t3 = time.perf_counter()
        times_heap.append(t3 - t2)

    return (sum(times_array)/len(times_array),
            sum(times_heap)/len(times_heap),
            sum(ms)/len(ms))

def experiment_comparision():
    sizes = [80, 120, 160, 200, 240, 280, 320]
    ps = [0.05, 0.1, 0.2, 0.4]
    rows = []
    for p in ps:
        for n in sizes:
            mean_array, mean_heap, m_avg = time_comparision(n, p)
            rows.append({
                "n": n,
                "p": p,
                "m": int(m_avg),
                "n2": n*n,
                "n2_plus_m": n*n + int(m_avg),
                "time_array": mean_array,
                "time_heap": mean_heap
            })
    return rows

def plot_comparision(rows):
    plt.figure()
    ps = sorted({r["p"] for r in rows})
    for p in ps:
        sub = sorted([r for r in rows if r["p"] == p], key=lambda r: r["n"])
        xs = [r["n"] for r in sub]
        ys_array = [r["time_array"] for r in sub]
        ys_heap = [r["time_heap"] for r in sub]
        plt.plot(xs, ys_array, marker="o", label=f"Array, p={p}")
        plt.plot(xs, ys_heap, marker="x", label=f"Heap, p={p}")
    plt.xlabel("|V|")
    plt.ylabel("Mean time (s)")
    plt.title("Array VS Heap Dijkstra")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Array_VS_Heap.png")
    plt.close()


result = experiment()
plot_time_vs_n(result)
plot_time_vs_n2(result)
plot_time_vs_n2_plus_m(result)

#C
plot_comparision(experiment_comparision())
