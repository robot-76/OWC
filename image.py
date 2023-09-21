import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

image = Image.open(r'F:\EEE\EEE BOOKS\EEE Books -41\Thesis 4000\report\image\1500hzmanframe3.png')
image1 = ImageOps.grayscale(image)
img_array = np.array(image1, dtype=np.uint8)
image1.show()
print("image",img_array.shape)
print("zeros",img_array[:, 65])
t = np.arange(0, 1080, 1)
n = img_array[:, 70]
print("n=",len(n))
print("t=",len(t))
ave = (np.sum(n)/len(n)) + 2
new = [new - ave for new in n]
final = []
for i in range(len(new)):
    if abs(new[i]) > 10:
        final = np.append(final, new[i])
    else:
        final = np.append(final, 0)
ct = 0
zt = 0
z = []
on = []
oz = []
idx = []
for j, data in enumerate(final):
    if abs(data != 0):
        if zt > 0:
            z = np.append(z, zt)
            oz = np.append(oz, zt)
            idx = np.append(idx, j)
        zt = 0
        ct = ct + 1
    elif data == 0:
        if ct > 0:
            on = np.append(on, ct)
            oz = np.append(oz, ct)
            idx = np.append(idx, j)
        ct = 0
        zt = zt + 1
print(z)

print(on)
print(oz)
print(idx)
print(ave)
print(new)
print(final)

der = np.gradient(n, t)
plt.figure()
plt.subplot(211)
plt.plot(t, n)
plt.xlabel('Row index')
plt.ylabel('Normalized intensity')
plt.subplot(212)
plt.plot(t, final)
plt.xlabel('Row index')
plt.ylabel('Normalized intensity after thresholding')
# plt.plot(t, der, 'r', t, n, 'g')
plt.show()
# peaks, _ = find_peaks(img_array[:, 65], height=0)
# plt.plot(x)
# plt.plot(peaks, x[peaks], "x")
# plt.plot(np.zeros_like(x), "--", color="gray")
# plt.show()
# for j in range(len(n)):
#     if 400 <= j <= 600:
#         n[j] = n[j]+9
#     elif 600 < j <= 720:
#         n[j] = n[j] + 7
#     elif 300 <= j < 400:
#         n[j] = n[j] + 5
#     elif 0 <= j <= 50:
#         n[j] = n[j] + 5
#     else:
#         n[j] = n[j]
# no = []
# for k in range(len(n)):
#     if k % 20 == 0:
#         no = np.append(no, n[k])
# t1 = np.arange(0, 720, 20)
# plt.subplot(212)
# plt.plot(t, n, 'k')
# b1 = n - 23.5
# b2 = no - 23.5
# plt.figure()
# plt.subplot(211)
# plt.plot(t1, b2, 'bo', t, b1, 'k')
# bit = []
# for i in range(len(b2)):
#     if b2[i] >= 0:
#         bit = np.append(bit, 1)
#     else:
#         bit = np.append(bit, 0.05)
# print(bit)
# t2 = np.arange(0, len(bit), 1)
# plt.subplot(212)
# plt.bar(t2, bit)
# plt.show()
data = [1, 1, 1, 0.5, 1, 1, 0.5, 1, 1, 0.5, 0.5]

bar = np.arange(len(data))

fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(bar, data, color='b',
        width=0.6)

plt.xlabel("Bit Labelling")
plt.ylabel("Binary Bit")
plt.xticks([r for r in range(len(data))],
        ['H', 'H', 'H', 'Bit0', 'Bit1', 'Bit1', 'Bit0', 'Bit1', 'Bit1', 'Bit0', 'Bit0'])
plt.show()
