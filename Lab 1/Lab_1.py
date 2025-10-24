import random
import matplotlib.pyplot as plt
from time import time

n = 0

def merge(lst1, lst2):
    global n
    i = j = 0
    out = []
    while i < len(lst1) and j < len(lst2):
        n += 1
        if lst1[i] <= lst2[j]:
            out.append(lst1[i])
            i+=1
        else:
            out.append(lst2[j])
            j += 1
    out.extend(lst1[i:])
    out.extend(lst2[j:])
    return out

def hybridsort(lst, S):

    if len(lst) <= S:
        return insertionsort(lst)

    mid = len(lst)//2

    return merge(hybridsort(lst[:mid], S), hybridsort(lst[mid:], S))

def insertionsort(lst):
    global n
    for i in range(1, len(lst)):
        a = i-1
        item = lst[i]
        while a >= 0:
            n+= 1
            if lst[a] > item:
                lst[a+1] = lst[a]
                a-=1
            else:
                break
        lst[a+1] = item
    return lst

def merge_sort(A):
    if len(A) <= 1:
        return A

    mid = len(A) // 2
    B1 = merge_sort(A[:mid])
    B2 = merge_sort(A[mid:])
    return merge(B1, B2)

fixed_S_times = []
fixed_S_comparisons = []

fixed_list_times = []
fixed_list_comparisons = []
for i in range(3, 7):
    lst = [random.randint(1, 10**i) for j in range(10**i)]
    n = 0
    t = time()
    hybridsort(lst, 16)
    fixed_S_times.append(time()-t)
    fixed_S_comparisons.append(n)
    
print(fixed_S_times) #n = 1000, 10000, 100_000, 1_000_000 Time S = 16
print(fixed_S_comparisons) # comps

# plot graph
sizes = [10**i for i in range(3,7)]
plt.figure()
plt.plot(sizes, fixed_S_comparisons, marker='o')
plt.xlabel("Input size (n)")
plt.ylabel("Key Comparisons")
plt.title("Hybrid Sort (S=16)")
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.savefig("hybrid_sort_fixed_S.jpg", dpi=300)
plt.close()
plt.show()


print("------------100,000----------")
for i in range(5,101):
    lst = [random.randint(1, 10**5) for j in range(10**5)]
    n = 0
    t = time()
    hybridsort(lst, i)
    fixed_list_times.append(time()-t)
    fixed_list_comparisons.append(n)

print(fixed_list_times)  # n fixed to 100,000, time for different s
print(fixed_list_comparisons)  # comps for different s
S_values = list(range(5, 101))

idx = fixed_list_times.index(min(fixed_list_times))
print("Lowest S is:", S_values[idx], "with time =", fixed_list_times[idx])


plt.figure()
plt.plot(S_values, fixed_list_times, marker='o')
plt.xlabel("Threshold S")
plt.ylabel("Key Comparisons")
plt.title("Hybrid Sort where n is at 100,000")
plt.grid(True)
plt.savefig("hybrid_sort_fixed_list_comps.jpg", dpi=300)
plt.close()
plt.show()

for i in range(4):
    start = i * 25
    end = min((i + 1) * 25, len(S_values))
    s_chunk = S_values[start:end]
    t_chunk = fixed_list_times[start:end]
    if not s_chunk:
        continue
    plt.figure()
    plt.plot(s_chunk, t_chunk, marker='o')
    plt.xlabel("Threshold S")
    plt.ylabel("Time (s)")
    plt.title(f"Hybrid Sort (n=100,000) — Time vs S [{s_chunk[0]}–{s_chunk[-1]}]")
    plt.grid(True)
    plt.savefig(f"hybrid_sort_fixed_list_time(100,000)_part_{i + 1}_of_4.jpg", dpi=300)
    plt.close()

fixed_list_times = []
fixed_list_comparisons = []
for i in range(5,101):
    lst = [random.randint(1, 10**6) for j in range(10**6)]
    n = 0
    t = time()
    hybridsort(lst, i)
    fixed_list_times.append(time()-t)
    fixed_list_comparisons.append(n)

print("------------1,000,000----------")
print(fixed_list_times) # n fixed to 1,000,000, time for different s
print(fixed_list_comparisons) # comps for different s

idx = fixed_list_times.index(min(fixed_list_times))
print("Lowest S is:", S_values[idx], "with time =", fixed_list_times[idx])

S_values = list(range(5,101))
plt.figure()
plt.plot(S_values, fixed_list_comparisons, marker='o')
plt.xlabel("Threshold S")
plt.ylabel("Key Comparisons")
plt.title("Hybrid Sort where n is at 1,000,000")
plt.grid(True)
plt.savefig("hybrid_sort_fixed_list_comps.jpg", dpi=300)
plt.close()
# plt.show()

for i in range(4):
    start = i * 25
    end = min((i+1) * 25,len(S_values))
    s_chunk = S_values[start:end]
    t_chunk = fixed_list_times[start:end]
    if not s_chunk:
        continue
    plt.figure()
    plt.plot(s_chunk, t_chunk, marker='o')
    plt.xlabel("Threshold S")
    plt.ylabel("Time (s)")
    plt.title(f"Hybrid Sort (n=1,000,000) — Time vs S [{s_chunk[0]}–{s_chunk[-1]}]")
    plt.grid(True)
    plt.savefig(f"hybrid_sort_fixed_list(1,000,000)_time_part_{i + 1}_of_4.jpg", dpi=300)
    plt.close()

S_values = list(range(5,101))
fixed_list_times = []
fixed_list_comparisons = [] 
for i in range(5,101):
    lst = [random.randint(1, (10 ** 5)) for j in range((10 ** 5))]
    n = 0
    t = time()
    hybridsort(lst, i)
    fixed_list_times.append(time()-t)
    fixed_list_comparisons.append(n)
idx = fixed_list_times.index(min(fixed_list_times))
print("Lowest S is:", S_values[idx], "with time =", fixed_list_times[idx])




plt.figure(figsize=(20, 5))
plt.plot(S_values, fixed_list_times, marker='o')
plt.xlabel("Threshold S")
plt.ylabel("Times")
plt.title("Hybrid Sort where n is at 1,000")
plt.grid(True)
plt.savefig("hybrid_sort_fixed_list_times_10k.jpg", dpi=300)
plt.close()
idx = fixed_list_times.index(min(fixed_list_times))
print("Lowest S is:", S_values[idx], "with time =", fixed_list_times[idx])

lst = [random.randint(1, 10**5) for j in range(10**7)]
t = time()
n = 0
merge_sort(lst)
merge_time = time() - t
merge_comps = n
print("MergeSort Time:", merge_time)
print("MergeSort Comparisons:", merge_comps)

lst = [random.randint(1, 10**5) for j in range(10**7)]
t = time()
n = 0
S = 12
hybridsort(lst, S)
hybrid_time = time() - t
hybrid_comps = n
print("HybridSort Time:", hybrid_time)
print("HybridSort Comparisons:", hybrid_comps)

plt.figure()
plt.bar(["MergeSort", f"HybridSort S={S}"], [merge_time, hybrid_time], color=['blue', 'orange'])
plt.ylabel("Time (s)")
plt.title("Time Comparison on 10M integers")
plt.grid(axis='y')
plt.savefig("time_comparison.jpg", dpi=300)
plt.close()
plt.show()

# Plot Key Comparisons Comparison
plt.figure()
plt.bar(["MergeSort", f"HybridSort S={S}"], [merge_comps, hybrid_comps], color=['blue', 'orange'])
plt.ylabel("Key Comparisons")
plt.title("Key Comparisons Comparison on 10M integers")
plt.grid(axis='y')
plt.savefig("comparisons_comparison.jpg", dpi=300)
plt.close()
plt.show()

fastests = []
fastests_index = []
S_values = list(range(5,101))
for _ in range(100):
    fixed_list_times = []
    fixed_list_comparisons = [] 
    for i in range(5,101):
        lst = [random.randint(1, (10 ** 5)) for j in range((10 ** 5))]
        n = 0
        t = time()
        hybridsort(lst, i)
        fixed_list_times.append(time()-t)
        fixed_list_comparisons.append(n)
    idx = fixed_list_times.index(min(fixed_list_times))
    print("Lowest S is:", S_values[idx], "with time =", fixed_list_times[idx])
    
    fastests.append(S_values[idx])
    fastests_index.append(idx)
print("Average S:", sum(fastests_index)/len(fastests_index))

print("end")
