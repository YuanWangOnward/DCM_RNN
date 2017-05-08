
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(list(range(10)))
plt.plot(list(range(10)), '--')
plt.title('x true and hat Iteration = ' + str(1))