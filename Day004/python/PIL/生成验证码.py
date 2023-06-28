import random
import numpy as np
from PIL import Image,ImageDraw,ImageFont

class GenerateCoder:
    #生成随机内容(A-Z)
    def get_text(self):
        return  chr(random.randint(65,90))
    #生成随机字体颜色
    def font_color(self):
        return (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    def bg_color(self):
        return (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    #生成画板
    def back_ground(self):
        w,h=240,60
        pannel=Image.new(size=(w,h),color=(255,255,255),mode="RGB")
        #创建画笔
        draw=ImageDraw.Draw(pannel)
        #创建字体
        font=ImageFont.truetype(font="C:/Windows/Fonts/Arial.ttf",size=30)
        #画板上色
        for y in range(h):
            for x in range(w):
                draw.point((x,y),fill=self.bg_color())
        #填入验证码
        for i in range(4):
            draw.text((60*i+20,15),text=self.get_text(),fill=self.font_color(),font=font)
        #也可以使用numpy构建3维图填充白色得到画板
        return  pannel

if __name__ == '__main__':
    gen=GenerateCoder()
    img=gen.back_ground()
    img.show()
