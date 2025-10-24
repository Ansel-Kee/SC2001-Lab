def UnboundedKnapsack(C, n, w, p):
    P = [0 for i in range(C)]

    for c in range(0, C):
        for i in range(n):
            if w[i] <= c+1:
                P[c] = max(P[c], P[c - w[i]] + p[i])
    return P[C-1]

w = [4, 6, 8]
p = [7, 6, 9]
print(f'4a) {UnboundedKnapsack(14, len(w), w, p)}')
w = [5, 6, 8]
p = [7, 6, 9]
print(f'4b) {UnboundedKnapsack(14, len(w), w, p)}')