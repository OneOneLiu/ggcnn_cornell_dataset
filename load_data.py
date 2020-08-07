import os
import glob
import cv2
import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Load cornell dataset')
    
    parser.add_argument('--cornell_path', type=str, default='cornell', help='the path of cornell dataset')
    
    args = parser.parse_args()
    return args

args = parse_args()
cornell_path = args.cornell_path
print(cornell_path)
graspf = glob.glob(os.path.join(cornell_path,'*','pcd*cpos.txt'))
graspf.sort()

rgbf = [filename.replace('cpos.txt','r.png') for filename in graspf]
depthf = [filename.replace('cpos.txt','d.tiff') for filename in graspf]

def str2num(point):
    '''
    :功能  :将字符串类型存储的抓取框脚点坐标取整并以元组形式返回
    
    :参数  :point,字符串，以字符串形式存储的一个点的坐标
    :返回值 :列表，包含int型抓取点数据的列表[x,y]
    '''
    x,y = point.split()
    x,y = int(round(float(x))),int(round(float(y)))
    
    return (x,y)

def get_rectangles(cornell_grasp_file):
    '''
    :功能  :从抓取文件中提取抓取框的坐标信息
    
    :参数  :cornell_grap_file:字符串，指向某个抓取文件的路径
    :返回值 :列表，包含各个抓取矩形数据的列表
    '''
    grasp_rectangles = []
    with open(cornell_grasp_file,'r') as f:
        while True:
            grasp_rectangle = []
            point0 = f.readline().strip()
            if not point0:
                break
            point1,point2,point3 = f.readline().strip(),f.readline().strip(),f.readline().strip()
            grasp_rectangle = [str2num(point0),
                               str2num(point1),
                               str2num(point2),
                               str2num(point3)]
            grasp_rectangles.append(grasp_rectangle)
    
    return grasp_rectangles

def draw_rectangles(img_path,grasp_path):
    '''
    :功能  :在指定的图片上绘制添加相应的抓取标注框
    
    :参数  :img_path:字符串，指向某个RGB图片的路径
    :参数  :grasp_path:字符串，指向某个抓取文件的路径
    :返回值 :numpy数组，已经添加完抓取框的img数组
    '''
    img_path = img_path
    grasp_path = grasp_path
    
    img = cv2.imread(img_path)
    grs = get_rectangles(grasp_path)
    
    for gr in grs:
        #产生随机颜色
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        #绘制添加矩形框
        for i in range(3): #因为一个框只有四条线，所以这里是3
            img = cv2.line(img,gr[i],gr[i+1],color,2)
        img = cv2.line(img,gr[3],gr[0],color,2) #补上最后一条封闭的线
    
    cv2.imshow('img',img)
    cv2.waitKey(100)
    
    return img

if __name__ == "__main__":
    img = draw_rectangles(rgbf[500],graspf[500])