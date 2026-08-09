import matplotlib.pyplot as plt

weights = []

w = 10
lr = 0.1

for i in range(30):

    grad = 2*w

    w = w - lr*grad

    weights.append(w)

plt.plot(weights, marker="0")
plt.title("Weight During Gradient Descent")
plt.xlabel("Iteration")
plt.ylabel("Weight")
plt.grid(True)
plt.show()
