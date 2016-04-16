import cv2
import numpy as np
import matplotlib.pyplot as plt

#把圖片直接轉成灰階
img = cv2.imread('D:\\Freddy\\vision\\mp2\\think.jpg',0)
img2 = cv2.imread('D:\\Freddy\\vision\\mp2\\view.jpg',0)

#開始作傅立葉轉換
f = np.fft.fft2(img)
g = np.fft.fft2(img2)
fshift = np.fft.fftshift(f)
gshift = np.fft.fftshift(g)

#abs是取magnitude spectrum
#angle是取phase spectrum
woman_m = np.abs(fshift)
woman_p = np.angle(fshift)
rectangle_m = np.abs(gshift)
rectangle_p = np.angle(gshift)

img_new1_f = np.zeros(img.shape,dtype=complex) 
#real為實部的部分
#imag為虛部的部分
img1_real = woman_m*np.cos(rectangle_p) 
img1_imag = woman_m*np.sin(rectangle_p) 
img_new1_f.real = np.array(img1_real) 
img_new1_f.imag = np.array(img1_imag) 
f3shift = np.fft.ifftshift(img_new1_f) 
img_new1 = np.fft.ifft2(f3shift)
img_new1 = np.abs(img_new1)
#調整圖片大小來顯示
img_new1 = (img_new1-np.amin(img_new1))/(np.amax(img_new1)-np.amin(img_new1))
plt.subplot(244),plt.imshow(img_new1,'gray'),plt.title('think_m&view_p')
plt.xticks([]),plt.yticks([])


img_new2_f = np.zeros(img2.shape,dtype=complex) 
#real為實部的部分
#imag為虛部的部分
img2_real = rectangle_m*np.cos(woman_p) 
img2_imag = rectangle_m*np.sin(woman_p) 
img_new2_f.real = np.array(img2_real) 
img_new2_f.imag = np.array(img2_imag) 
f4shift = np.fft.ifftshift(img_new2_f) 
img_new2 = np.fft.ifft2(f4shift)
img_new2 = np.abs(img_new2)
#調整圖片大小來顯示
img_new2 = (img_new2-np.amin(img_new2))/(np.amax(img_new2)-np.amin(img_new2))
plt.subplot(248),plt.imshow(img_new2,'gray'),plt.title('think_p&view_m')
plt.xticks([]),plt.yticks([])




#印出男人思考的原圖,magnitude spectrum,phase spectrum
plt.subplot(241),plt.imshow(img, cmap = 'gray')
plt.title('Think'), plt.xticks([]), plt.yticks([])
plt.subplot(242),plt.imshow(woman_m, cmap = 'gray')
plt.title('Think Magnitude'), plt.xticks([]), plt.yticks([])
plt.subplot(243),plt.imshow(woman_p, cmap = 'gray')
plt.title('Think Phase'), plt.xticks([]), plt.yticks([])
#印出風景的原圖,magnitude spectrum,phase spectrum
plt.subplot(245),plt.imshow(img2, cmap = 'gray')
plt.title('View'), plt.xticks([]), plt.yticks([])
plt.subplot(246),plt.imshow(rectangle_m, cmap = 'gray')
plt.title('View Magnitude'), plt.xticks([]), plt.yticks([])
plt.subplot(247),plt.imshow(rectangle_p, cmap = 'gray')
plt.title('View Phase'), plt.xticks([]), plt.yticks([])
plt.show()