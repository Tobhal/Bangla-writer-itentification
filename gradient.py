import numpy as np
import cv2
from PIL import Image
from math import sqrt
from matplotlib import pyplot as plt
from itertools import chain
import cv2

from Util import h_from_image, greatest, least

file = 'Bangla-writer-test/test_001.tif'
import math
img = Image.open(file)
print(img.size)
img = np.asarray(img)
img = 255.0 - img
print(img.shape)
print(img[0])
img = img /255.0
plt.imshow(img, cmap='Greys')
plt.show()


def division(k,a,dim):
    j,prev = 0,0
    p = k/dim
    print('IN division function',p)
    for i in range(dim):
        j = j+p
        q = j
        t = q+0.5
        if j>t:
            a[i] = q+1-prev
        else:
            a[i] = q-prev
        prev = prev + a[i]
    return a


row,col = img.shape
print(img.shape)
c1,c2,c3,c4 = np.zeros((300,)),np.zeros((300,)),np.zeros((300,)),np.zeros((300,))
l1,l2,l3,l4,=0,0,0,0
d = np.zeros((9,9,16))
nc1 = np.zeros((11,11,16))
nc2 = np.zeros((5,5,16))
appe = np.zeros((400,))
print(appe.shape)
for i in range(300):
    c1[i],c2[i],c3[i],c4[i] = 999,0,999,0
u = col-1
y1 = row-1



h1,h2,h3,h4 = h_from_image(img)
ro = h4-h3+1
co = h2-h1+1
print(h1,h2,h3,h4)


counter = 0
for i in range(h3,h4+1):
    for j in range(h1,h2+1):
        if img[i][j]==1:
            counter = counter+1
print('counter is:',counter)
for k in range(1,6):
    for i in range(h3,h4+1):
        for j in range(h1,h2+1):
            s = 0
            s = s+img[i][j]
            if i>=h3 and i<=h4 and j+1>=h1 and j+1<=h2:
                s=s+img[i][j+1]
            if i+1>=h3 and i+1<=h4 and j>=h1 and j<=h2:
                s=s+img[i+1][j]
            if i+1>=h3 and i+1<=h4 and j+1>=h1 and j+1<=h2:
                s=s+ img[i+1][j+1];
            img[i][j]=s;

total = 0
for i in range(h3,h4+1):
    for j in range(h1,h2+1):
        total = total+img[i][j]
count = ro*co
mean = total/count
print('count is:',ro*co,' total is:',total)
print('mean is: ',mean)
for i in range(h3,h4+1):
    for j in range(h1,h2+1):
        img[i][j] = img[i][j] - mean




maxi = img[h3][h1]
for i in range(h3,h4+1):
    for j in range(h1,h2+1):
        if maxi<img[i][j]:
            maxi = img[i][j]
print('maxx is :',maxi)

for i in range(h3,h4+1):
    for j in range(h1,h2+1):
        img[i][j] = img[i][j]/maxi


r11,c11 = np.zeros((20,)),np.zeros(20,)
r11 = division(ro,r11,9)
c11 = division(co,c11,9)
img1 = []


for i in range(h3,h4+1):
    img1_row = []
    for j in range(h1,h2+1):
        img1_row.append(img[i][j])
    img1.append(img1_row)
img1 = np.asarray(img1)

print(' shape of img1 is:',img1.shape)
co = co-1
print(ro,co,' ro and co is')
for i in range(ro):
    img1[i][co-1] = (img1[i][co-1]+ img1[i][co-2])/2
for j in range(co):
    img1[ro-1][j]=(img1[ro-1][j]+ img1[ro-2][j])/2

img1[ro-1][co-1]=(img1[ro-1][co-1]+img1[ro-2][co-2])/2

for m in range(0,9):
    for n in range(0,9):
        s1=0
        t=0
        y,z=0,0
        for i in range(0,m):
            y=y+r11[i]

        z=y+r11[m]
        for j in range(0,n):
            s1=s1+c11[j]

        t=s1+c11[n]
        y,z,s1,t = int(y),int(z),int(s1),int(t)
        for i in range(y,z-1):
            for j in range(s1,t-1):
                fxy=0
                angxy=0
                delu=0
                delv=0
                delu=img1[i+1][j]-img1[i][j]
                delv=img1[i][j+1]-img1[i][j]
                if delu==0:
                    continue
                else:
                    fxy=sqrt((delu*delu)+(delv*delv))
                    angxy=(math.atan2(delv,delu)*180)/3.142857
                    angxy=angxy+180
                    rxy=angxy/22.5
                    rxy = int(rxy)
                    #fxy = int(fxy)
                    d[m][n][rxy]=d[m][n][rxy]+fxy

i,j=0,0
while i<9:
    while j<9:
        for k in range(0,16,1):
            d[i][j][k] = d[i][j][k]*2
        j=j+2
    i=i+2

for i in range(1,10,1):
    for j in range(1,10,1):
        for k in range(0,16,1):
            #print(d[i-1][j-1][k])
            nc1[i][j][k] = d[i-1][j-1][k]

for i in range(1,10,2):
    for j in range(1,10,2):
        for k in range(0,16,1):
            nc2[int(i/2)][int(j/2)][k]=nc1[i][j][k]+nc1[i-1][j-1][k]+nc1[i-1][j][k]+nc1[i-1][j+1][k]+nc1[i][j-1][k]+nc1[i][j+1][k]+nc1[i+1][j-1][k]+nc1[i+1][j][k]+nc1[i+1][j+1][k]


h11 = 0
for i in range(0,5):
    for j in range(0,5):
        for k in range(0,16):
            appe[h11] = nc2[i][j][k]/counter
            h11 = h11+1

print(appe)

for i in appe:
    print(f'{i},', end='')