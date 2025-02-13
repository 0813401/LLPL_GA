import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def genecoef(gene):
  tc = (np.sum(2**np.arange(4) * gene[:4]) + 500) # 0 ~ 15 => 500 ~ 515 
  beta = (np.sum(2**np.arange(12) * gene[4:16]) + 1300) / 10000 # 0 ~ 4095 => 0.13 ~ 0.5395 
  omega = (np.sum(2**np.arange(12) * gene[16:28]) + 4400) / 1000 # 0 ~ 4095 => 4.4 ~ 8.495
  phi = (np.sum(2**np.arange(10) * gene[28:])) / 100 # 0.00 ~ 10.24 => 0.00 ~ 6.28
  while(phi >= 6.28):
    for i in range(28, 38):
      gene[i] = np.random.randint(0, 2)
    phi = (np.sum(2**np.arange(10) * gene[28:])) / 100
  return tc, beta, omega, phi

def LLPL_function(A, B, C, tc, t, beta, omega, phi):
  return A + B * (tc - t)**beta * (1 + C * np.cos(omega * np.log(tc - t) + phi))

def MAE(real, predict):
  print(np.mean(abs(real[:len(predict)] - predict)))

# load the data
t = np.load("data.npy") # p(t)
# A 是 Amplitude 的 shift term
# B 是 increasing rate of (tc - t)^β
# tc 是泡沫爆掉的時間
# t 是現在的時間
# C, ω, Φ 控制週期的疏密

# initial input
n = 8000
survival_rate = 0.01
survival_n = round(n * survival_rate)
mutation = round(n * 38 * 0.001) # 有幾個基因會突變 0.001 -> 0.1%
population = np.random.randint(0, 2, (n, 38)) # 產生0或1(bits)的gene序列組(10000人, 每個人有4個gene(10(bits), 4, 4, 10))
fit = np.zeros((n, 1))

# 繁衍10次
for G in range(6):
  print("G:",G)
  for i in range(n):
    tc, beta, omega, phi = genecoef(population[i, :])
    a = np.zeros((tc, 3))
    b = np.zeros((tc, 1))
    time = np.arange(tc)
    for row in range(tc):
      a[row, :] = [1, (tc - row)**beta, (tc - row)**beta * np.cos(omega * np.log(tc - row) + phi)]
      b[row, :] = np.log(t[row])

    estimate_ABC = np.linalg.lstsq(a, b)[0]
    A = float(estimate_ABC[0])
    B = float(estimate_ABC[1])
    C = float(estimate_ABC[2]/estimate_ABC[1])
  
    tmp = LLPL_function(A, B, C, tc, time, beta, omega, phi) 
    fit[i] = np.mean(abs(t[:len(tmp)] - np.exp(1)**tmp))

  # 競爭式選擇 (tournament selection) - 只留fitness最高的一小群人survive，淘汰適應不佳的

  # 回傳由小到大值的index
  sortf = np.argsort(fit[:, 0])
  population = population[sortf, :]

  # 交配 (crossover) - 採用 => 遮罩交配 (產生一個0/1 mask或filter，mask為1的bit互換)
  # 從 100~9999 的爛基因要被取代了(By父母的交配)
  for i in range(survival_n, n):
    # 選出生存下那些人中，爸爸及媽媽的人選
    fid = np.random.randint(0, survival_n)
    mid = np.random.randint(0, survival_n)
    while fid == mid:
      mid = np.random.randint(0, survival_n)
    mask = np.random.randint(0, 2, (1, 38))
    # 先將媽媽的gene全部複製給兒子
    son = population[mid, :].copy()
    father = population[fid, :]
    # 讓兒子mask是1的地方，全部轉成爸爸mask是1的地方
    son[mask[0, :] == 1] = father[mask[0, :] == 1]
    population[i, :] = son

  # 突變 - 少數bit 0->1或1->0
  for i in range(mutation):
    m = np.random.randint(survival_n, n) 
    n_gene = np.random.randint(0, 38)
    population[m, n_gene] = 1 - population[m, n_gene]


for i in range(n):
  tc, beta, omega, phi = genecoef(population[i, :])
  time = np.arange(tc)
  b = np.log(t[:tc])
  tmp = LLPL_function(A, B, C, tc, time, beta, omega, phi) 
  fit[i] = np.mean(abs(t[:len(tmp)] - np.exp(1)**tmp))

sortf = np.argsort(fit[:, 0])
population = population[sortf, :]

tc, beta, omega, phi = genecoef(population[0, :])
time = np.arange(tc)

print("A =", A)
print("B =", B)
print("C =", C)
print("tc =", tc)
print("beta =", beta)
print("omega =", omega)
print("phi =", phi)

for i in range(tc):
  ans = np.exp(1)**LLPL_function(A, B, C, tc, time, beta, omega, phi) 

# 答案的outpUt
print('e = ',end="")
MAE(t, ans)

plt.plot(ans, color = 'orange')
plt.plot(t, color = 'green')
plt.show()