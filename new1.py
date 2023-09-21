
import cv2
import numpy as np
from itertools import groupby
print(cv2.__version__)
vidcap = cv2.VideoCapture('F:/EEE/EEE BOOKS/EEE Books -41/Thesis 4000/image/frame27.mp4')
success,image = vidcap.read()
count = 0
success = True
frames = []
while success:
    frames.append(image)
    # cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file if you want
    success,image = vidcap.read()
    # print('Read a new frame: ', success)
    count += 1
print(count, " frames extracted")
frames = np.array(frames)
print("data shape =\t", frames.shape)

# downsample
from scipy import ndimage
ds_frames = ndimage.zoom(frames,(1., .4, .4, 1.))
print("downsampled shape =\t", ds_frames.shape)
np.save("frames_data.npy", ds_frames)
print("first frame =\t", ds_frames[1])
cv2.imshow("frame 0",ds_frames[0])
cv2.imshow("frame 1",ds_frames[1])
cv2.imshow("frame 2",ds_frames[2])
cv2.imshow("frame 3",ds_frames[3])
cv2.imshow("frame 4",ds_frames[4])

print(len(ds_frames))
binaryOfText=[]
manchesterCode=[]
for i in range(len(ds_frames)):
    gimg = cv2.cvtColor(ds_frames[i], cv2.COLOR_BGR2GRAY)
    new_gimg = np.zeros(gimg.shape, gimg.dtype)
    contrast = 5.0
    bright = 4
    for y in range(gimg.shape[0]):  # Enhance contrast and brightness
        for x in range(gimg.shape[1]):
            new_gimg[y, x] = np.clip(contrast * gimg[y, x] + bright, 0, 255)

    ret, threshimg = cv2.threshold(new_gimg, 190, 255, cv2.THRESH_BINARY)  # use it Global Thresholdig or binarize image
    athreshimg = cv2.adaptiveThreshold(new_gimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)  # Adaptive Thresholdig or binarize image

    # value=np.array(threshimg)
    # sumofcolumn=np.sum(value,axis=0)
    # rowv=sumofcolumn[:]
    # sumofrow=athreshimg.sum(axis =1)
    # normalizerow=rowv/sumofrow[:, np.newaxis]
    # print("row value=",rowv)
    # print("value=",value)
    # print(value.shape)
    # print("nval",normalizerow)

    kernel = np.ones((5, 5), np.uint8)
    erodedImag = cv2.erode(threshimg, kernel, iterations=1)
    erodevalue = np.array(erodedImag)  # convert image into matrix
    sumervrow = np.sum(erodevalue, axis=1)  # sum each pixel of row
    print("erodeVal", erodevalue)
    print("esumcol", sumervrow)

    # represent black strip row of pixel by binary 0 & white strip row of pixel by binary 1
    vector = []
    for k in sumervrow:
        if k == 0:
            vector.append(int(0))
        else:
            vector.append(int(1))
    rowBinary = vector
    print("rowBinary", rowBinary)


    # count black strip row no. and white strip row no
    def consecutive_length(rowBinary):
        def sub(idx, lst, last_char, count):
            try:
                c = rowBinary[idx]  # c will be the 'next' char
            except IndexError:  # no more chars left to process
                if count:
                    lst.append(count)
                return lst
            if c != last_char:
                lst.append(count)
                count = 0
            return sub(idx + 1, lst, c, count + 1)

        return sub(0, [], rowBinary[0] if rowBinary else None, 0)


    print(consecutive_length(rowBinary))
    stripwidth = consecutive_length(rowBinary)
    lengthOfRow = len(stripwidth)
    print("rowlength", lengthOfRow)  # total no of strip

    # separate black and white strip and count strip no. in each color
    # separate odd and even value
    oddValue = []
    evenValue = []
    for i in range(len(stripwidth)):
        if (i % 2 == 0):
            evenValue.append(stripwidth[i])
        else:
            oddValue.append(stripwidth[i])
    print("oddvalue", oddValue)
    print("evenvalue", evenValue)


    def max_repeated_value(lst):
        max_count = 0
        max_value = None
        for value, group in groupby(lst):
            count = len(list(group))
            if count > max_count:
                max_count = count
                max_value = value
        return max_value, max_count


    EminVal, maxCount = max_repeated_value(evenValue)

    print("Eminval", EminVal)
    eroundVal = []
    for j in evenValue:
        if (round(j/EminVal))>=2:
         eroundVal.append(2)
        else:
            eroundVal.append(1)
    eroundValue = eroundVal
    print("eroundVal", eroundValue)  # white or black strip no.
    print("sum of eroundVal", sum(eroundValue))

    if len(stripwidth)>1:
      omniVal,maxCount = max_repeated_value(oddValue)

      print("ominval", omniVal)
      oroundVal = []
      for j in oddValue:
          if (round(j/omniVal))>=2:
           oroundVal.append(2)
          else:
              oroundVal.append(1)
      oroundValue = oroundVal
      print("oroundVal", oroundValue)  # black or white strip no.
      print("sum of oroundVal", sum(oroundValue))
    else:
        oroundValue=[]

    # find round value
    roundVal = []
    for j in range(len(eroundValue)):
        for i in range(len(oroundValue)):
          if (j ==i ):
            roundVal.append(eroundValue[j])
            roundVal.append(oroundValue[i])
    roundValue = roundVal
    print("roundVal", roundValue)  # row of black and white strip no.

    # represent black and white strip by binary digit that represent manchester code
    binaryDigit = []
    if rowBinary[0] == 0:
        for o in range(len(roundVal)):
            if (o % 2 == 0):
                for l in range(roundVal[o]):
                    binaryDigit.append(int(0))
            else:
                for l in range(roundVal[o]):
                    binaryDigit.append(int(1))
    else:
        for o in range(len(roundVal)):
            if (o % 2 == 0):
                for l in range(roundVal[o]):
                    binaryDigit.append(int(1))
            else:
                for l in range(roundVal[o]):
                    binaryDigit.append(int(0))
    manchesterCode.append(binaryDigit)
    print("manchester code", manchesterCode)
    print(len(manchesterCode))
    print(type(manchesterCode))

# add all manchester list in one frame manchester list
manchesterCVector=sum(manchesterCode, [])
print("Manchester c vector ",manchesterCVector)
print(type(manchesterCVector))

# Find the index of the first occurrence of 0
first_zero_index = manchesterCVector.index(0)

# Find the index of the second occurrence of 0
second_zero_index = manchesterCVector.index(0, first_zero_index + 1)

# Remove the first two occurrences of 0 using list slicing
manchesterCodeVector = manchesterCVector[:first_zero_index] + manchesterCVector[second_zero_index+1:]

print("Manchester c vector ",manchesterCodeVector)

# demodulation manchester code to binary value
evenman = []
oddman = []
l = len(manchesterCodeVector)
if (l % 2) == 0:
    for i in range(len(manchesterCodeVector)):
        if (i % 2) == 0:
            evenman.append(manchesterCodeVector[i])
        else:
            oddman.append(manchesterCodeVector[i])
else:
     for i in range(len(manchesterCodeVector) - 1):
        if (i % 2) == 0:
            evenman.append(manchesterCodeVector[i])
        else:
            oddman.append(manchesterCodeVector[i])
print("evenmanche", evenman)  # represent demodulated value if manchester code for 1 is 1 and 0 ; 0 is 0 and 1
print("oddmanche", oddman)  # represent demodulated value if manchester code for 1 is 0 and 1 ; 0 is 1 and 0
binaryOfText.append(oddman)


print("binaryOfText", binaryOfText)
print(type(binaryOfText))
# add all binary list in one frame binary list
binaryOfTextVector=sum(binaryOfText, [])
print("binary of text vector ",binaryOfTextVector)

# convert "binaryDigit" list to string
binaryString =''.join(map(str,binaryOfTextVector))
print("binaryString",binaryString)
print("string lenght",len(binaryString))
print(type(binaryString))

# convert binary to text or decode into text
input=binaryString
length = 8
input_l = [input[i:i+length] for i in range(0,len(input),length)]
print(input_l)
print(len(input_l))
outtext=[]
outdecimal=[]
for i in input_l:
  outdecimal.append(int(i, 2))
  outtext.append(chr(int(i, 2)))
print("outdecimal",outdecimal)
print("outtext",outtext)
output_Data = ''.join(outtext) # decoded output of image
print("Output Data: ",output_Data)
cv2.waitKey(0)
