import random
import time
import matplotlib.pyplot as plt
class MinHeap:
    def __init__(self):
        self.heap = []
        

    def push(self, val):
        self.heap.append(val)
        i = len(self.heap) - 1
        while i > 0 and self.heap[(i - 1) // 2] > self.heap[i]:
            self.heap[i], self.heap[(i - 1) // 2] = self.heap[(i - 1) // 2], self.heap[i]
            i = (i - 1) // 2
    def size(self):
        return len(self.heap)
    def delete(self, value):
        i = -1
        for j in range(len(self.heap)):
            if self.heap[j] == value:
                i = j
                break
        if i == -1:
            return
        self.heap[i] = self.heap[-1]
        self.heap.pop()
        while True:
            left = 2 * i + 1
            right = left + 1
            smallest = i
            if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
                smallest = right
            if smallest != i:
                self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
                i = smallest
            else:
                break

    def pop(self):
        out = self.heap[0]
        self.delete(out)
        return out

    def heapify(self, i, n):
        smallest = i
        left = 2 * i + 1
        right = left + 1

        if left < n and self.heap[left] < self.heap[smallest]:
            smallest = left
        if right < n and self.heap[right] < self.heap[smallest]:
            smallest = right
        if smallest != i:
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            self.heapify(smallest, n)


def dijkstra(V, adj, src):
    pq = MinHeap()
    
    dist = [float('inf')] * V

    pq.push([0, src])
    dist[src] = 0

    while pq.size():
        u = pq.pop()[1]

        for x in adj[u]:
            v, weight = x[0], x[1]

            if dist[v] > dist[u] + weight:
                dist[v] = dist[u] + weight
                pq.push([dist[v], v])

    return dist


def test(V, E):
    src = 0

    adj = [[] for _ in range(V)]
    pairs = {i:[j for j in range(V) if j != i] for i in range(V)}
    for i in range(E):
        start = random.choice(list(pairs.keys()))
        end = random.choice(pairs[start])
        weight = random.randint(1, 10)
        adj[start].append([end, weight])
        adj[end].append([start, weight])

        pairs[start].remove(end)
        pairs[end].remove(start)
        if len(pairs[start]) == 0:
            pairs.pop(start, None)
        if len(pairs[end]) == 0:
            pairs.pop(end, None)
    t = time.time()
    dijkstra(V, adj, src)
    t = time.time()-t
    return t

runs = 50

v_values = [200, 400, 800, 1600, 3200, 6400, 12800]
v_times = []
for i in v_values:
    avg = 0
    for _ in range(runs):
        avg += test(i, i*2)
    v_times.append(avg/runs)

plt.figure()
plt.plot(v_values, v_times, marker="o")
plt.xlabel("|V| (n)")
plt.ylabel("Mean time (s)")
plt.title("Dijkstra (array min): Time vs |V|")
plt.legend()
plt.tight_layout()
plt.savefig("time vs v (heap).png")
plt.close()

e_values = [999, 1500, 2000, 3000, 4000, 6400, 12800]
e_times = []
for i in e_values:
    avg = 0
    for _ in range(runs):
        avg += test(1000, i)
    e_times.append(avg/runs)

plt.figure()
plt.plot(e_values, e_times, marker="o")
plt.xlabel("|E| (n)")
plt.ylabel("Mean time (s)")
plt.title("Dijkstra (array min): Time vs |E|")
plt.legend()
plt.tight_layout()
plt.savefig("time vs e (heap).png")
plt.close()