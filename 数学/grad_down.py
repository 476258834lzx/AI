import random
import matplotlib.pyplot as plt

x=[i/100 for i in range(100)]
y=[3*e+4+random.random() for e in x]

w=random.random()
b=random.random()

plt.ion()

for i in range(30):
    for _x,_y in zip(x,y):
        z=w*_x+b
        o=  z-_y
        loss=o**2

        dw=-2*o*_x
        db=-2*o

        w=w+0.1*dw
        b=b+0.1*db

        v = [w * e + b for e in x]
        plt.cla()
        plt.plot(x, v)
        plt.plot(x, y, ".")
        plt.pause(0.01)
        print("w:",w,"b:",b,"loss:",loss)
plt.ioff()
plt.show()