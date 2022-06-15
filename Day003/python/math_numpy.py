import numpy as np
import matplotlib.pyplot as plt

x=np.arange(0.1,10,0.1)
y1=x*0+5
y2=x**2
y3=2**x
y4=np.log2(x)/np.log2(3)
y5=np.cos(x)

plt.plot(x,y1)#常函数
plt.plot(x,y2)#幂函数
plt.plot(x,y3)#指数函数
plt.plot(x,y4)#对数函数
plt.plot(x,y5)#三角函数
plt.show()