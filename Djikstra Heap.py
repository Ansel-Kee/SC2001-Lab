import random
import time
import matplotlib.pyplot as plt
import numpy as np

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

def dijkstra(V, adj):
    pq = MinHeap()
    
    dist = [float('inf')] * V

    pq.push([0, 0])
    dist[0] = 0

    while pq.size():
        pos, u = pq.pop()
        for x in adj[u]:
            v, weight = x[0], x[1]

            if dist[v] > dist[u] + weight:
                dist[v] = dist[u] + weight
                pq.push([dist[v], v])

    return dist

def make_graph(V, E):
    adj = [[] for _ in range(V)]
    unvisited = [i for i in range(V)]
    pairs = {i:[j for j in range(V) if j != i] for i in range(V)}
    for i in range(E-V):
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
        if start in unvisited:
            unvisited.remove(start)
        if end in unvisited:
            unvisited.remove(end)

    for i in unvisited:
        end = random.choice(pairs[i])
        weight = random.randint(1, 10)
        adj[i].append([end, weight])
        adj[end].append([i, weight])
        pairs[i].remove(end)
        pairs[end].remove(i)
        if len(pairs[i]) == 0:
            pairs.pop(i, None)
        if len(pairs[end]) == 0:
            pairs.pop(end, None)

    for _ in range(V-len(unvisited)):
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
    return adj

def test(V, E):

    adj = make_graph(V, E)
    t = time.time()
    dijkstra(V, adj)
    t = time.time()-t
    
    return t

runs = 50

v_values = [200, 400, 800, 1600, 3200, 6400, 12800]
v_times = []
for i in v_values:
    avg = 0
    for _ in range(runs):
        avg += test(i, (v_values[0]**2-v_values[0])//2)
    v_times.append(avg/runs)
    print(i)

logs = np.log(v_values).tolist()
func = [(v_values[i]+((v_values[0]**2-v_values[0])//2))*logs[i] for i in range(len(logs))]
print(func, len(func))
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel("|V| (n)")
ax1.set_ylabel("Mean time (s)")
ax1.plot(v_values, v_times, color=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.get_yaxis().set_visible(False)
ax2.plot(v_values, func, color=color)

fig.tight_layout()
plt.savefig("time vs v (heap) test.png")
plt.close()
runs = 200
e_values = [1000, 1500, 2000, 3000, 4000, 6400, 12800, 19900]
e_times = []
for i in e_values:
    avg = 0
    for _ in range(runs):
        avg += test(200, i)
    e_times.append(avg/runs)
    print(i)

logs = [np.log(200).tolist()]*len(e_values)
# print(logs)
func = [(200+e_values[i])*logs[i] for i in range(len(logs))]
print(func)
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel("|E| (n)")
ax1.set_ylabel("Mean time (s)")
ax1.plot(e_values, e_times, color=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.get_yaxis().set_visible(False)
ax2.plot(e_values, func, color=color)

fig.tight_layout()
plt.savefig("time vs e (heap) test.png")
plt.close()


