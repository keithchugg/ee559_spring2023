
import numpy as np 
import matplotlib.pyplot as plt

m = np.arange(0, 10, 0.1)
plt.figure()
plt.plot(m, 5 - 3 * m + 2 * m **2 + 1.5 * m **3, color='k', label=r'$5 - 3  m + 2 m^2 + 1.5 m^3$')
plt.plot(m,  (m ** 2), label=r'$m^2$')
plt.plot(m,  (m ** 3), label=r'$m^3$')
plt.plot(m, 2 * (m ** 3), label=r'$2 m^3$')
plt.plot(m, m ** 4, label=r'$m^4$')
plt.legend()
plt.grid(':')
plt.xlabel('m')
plt.ylim([0, 1200])
plt.xlim([0, 9])
plt.savefig('big_o.pdf')

plt.figure()
plt.plot(m, 5 - 3 * m + 2 * m **2 + 1.5 * m **3, color='k', label=r'$5 - 3  m + 2 m^2 + 1.5 m^3$')
plt.plot(m,  (m ** 2), label=r'$m^2$')
plt.plot(m,  (m ** 3), label=r'$m^3$')
plt.plot(m, 2 * (m ** 3), label=r'$2 m^3$')
plt.plot(m, m ** 4, label=r'$m^4$')
plt.legend()
plt.grid(':')
plt.xlabel('m')
plt.ylim([0, 10])
plt.xlim([0, 5])
plt.savefig('big_o_low_m.pdf')

