
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

import matplotlib.pyplot as plt

image_index = 7777 # Izbor nasumicnog broja cija je labela 8 zbog provere
print(y_train[image_index]) 
plt.imshow(x_train[image_index], cmap='Greys')



x_train.shape
input_shape=[]

def resize_region(region):
    #Transformisati selektovani region na sliku dimenzija 28x28
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)

def gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin
def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def transformisanjeInputa():
    
    # Vrsim reshape nad 4-dimenzionalnim nizom da bi mogao da radi sa Keras API
    global x_train
    global x_test
    global y_test
    global y_train
    global input_shape

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #Vrsimo normalizaciju RGB tako sto podelimo sa makslimlnom RGB vrednoscu

    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])


def kreiranjeModela():
    global input_shape
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Flattening the 2D arrays for fully connected layers
    model.add(Flatten()) 
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    return model


transformisanjeInputa()
#importovanje potrebnih keras modula koji sadrze model 
#i slojeve
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D    
model = kreiranjeModela()    


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
#model.fit(x=x_train,y=y_train, epochs=30)

from keras.models import load_model

#model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')


model.evaluate(x_test, y_test)



image_index = 4444
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())


import numpy as np
import cv2
import matplotlib.pyplot as plt

linesOdInteresa=[]


def detekcija(img):

    gray = gray_scale(frame)

    # primenjujem metodu za detekciju ivica
    edges = cv2.Canny(gray,75,150,apertureSize = 3) 
    
    # Dobijam niz od r i theta vrednosti 
    lines = cv2.HoughLines(edges,1,np.pi/180, 50) 
    

    # The below for loop runs till r and theta values 
    # are in the range of the 2d array 
    edges = cv2.Canny(gray, 75, 150)
 
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=20)
    
    global linesOdInteresa
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        f = ((x1 -x2)**2 + (y1-y2)**2)**(1/2)
        #147, 449, 398, 298
        #if(x1==147 and y1==449 and x2==398 and y2==298):
        if(f>150.00): #odaktativna procena samo duzine
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            linesOdInteresa.append([[x1,y1,x2,y2]])
            #print(x1, y1, x2, y2, f);

    #cv2.imshow("Edges", edges)
    #cv2.imshow("Image", img)
    cv2.imwrite("Edges.jpg", edges)
    cv2.imwrite("IMG.jpg", img)

nazivVidea = 'video-4.avi'    

cap = cv2.VideoCapture(nazivVidea)
ret, frame = cap.read()
#plt.imshow(frame)
detekcija(frame)

#Sklearn biblioteka sa implementiranim K-means algoritmom
from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd

def midpoint(x1, y1, x2, y2):
    return ((x1 + x2)/2, (y1 + y2)/2)

print(linesOdInteresa)
srednjetacke=[]

for line in linesOdInteresa:
    x1, y1, x2, y2 = line[0]
    #global srednjetacke
    srednjetacke.append(midpoint(x1,y1,x2,y2))
    
print(srednjetacke)    
A = np.squeeze(np.asarray(srednjetacke))
B=A[:,::2]
B.flatten() # pretvara matricu u niz
print(B.flatten())
C= A[:,1::2];
df = pd.DataFrame({
    'x':B.flatten(),
    'y':C.flatten()
})
kmeans = KMeans(n_clusters = 2,max_iter=2000, tol=0.00001, n_init=10)
kmeans.fit(df)
plt.imshow(frame)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_

fig = plt.figure()
colmap={1:'r',2:'g'}
colors = map(lambda x: colmap[x+1],labels)

colors1 = list(colors)
plt.scatter(df['x'],df['y'], color = colors1, alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlim(0,600)
plt.ylim(0,600)
plt.show()


prviNiz =[]
drugiNiz=[]
i =0
for x in labels:
    if x==0:
        prviNiz.append(linesOdInteresa[i])
    elif x==1:
        drugiNiz.append(linesOdInteresa[i])
    i=i+1
    
donja=[]
gornja=[]
if prviNiz[0][0][0] < drugiNiz[0][0][0]:
    donja = prviNiz
    gornja = drugiNiz
else:
    gornja = prviNiz
    donja = drugiNiz
print(donja)


donja = np.squeeze(np.asarray(donja))
print(donja)
gornja= np.squeeze(np.asarray(gornja))
print(gornja)

def line(x0, y0, x1, y1):
        "Bresenham's line algorithm"
        points_in_line = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points_in_line.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points_in_line.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points_in_line.append((x, y))
        return points_in_line


donja_sve_koord = []
gornja_sve_koord = []
for x,y,x1,y1 in donja:
    gornja_sve_koord.extend(line(x,y-5,x1,y1));

for x,y,x1,y1 in gornja:#pogresno sam pretpostavio koja je gornja a koja donja
    donja_sve_koord.extend(line(x,y-5,x1,y1));
    
#print(gornja_sve_koord)    
#for (x,y) in gornja_sve_koord:
#       cv2.putText(frame, '.',(x,y),cv2.FONT_HERSHEY_TRIPLEX, 1, (255,0,255)) 
#plt.imshow(frame)

#for (x,y) in donja_sve_koord:
#       cv2.putText(frame, '.',(x,y),cv2.FONT_HERSHEY_TRIPLEX, 1, (255,0,255)) 
#plt.imshow(frame)    


import operator

xmin,ymin=min(donja_sve_koord, key=operator.itemgetter(1))
xmax,ymax=max(donja_sve_koord, key=operator.itemgetter(1))
donja_sve_koord=line(xmin,ymin,xmax,ymax)
for (x,y) in donja_sve_koord:
    cv2.putText(frame, '.',(x,y),cv2.FONT_HERSHEY_TRIPLEX, 1, (255,0,255)) 

x1min,y1min=min(gornja_sve_koord, key=operator.itemgetter(1))
x1max,y1max=max(gornja_sve_koord, key=operator.itemgetter(1))
gornja_sve_koord=line(x1min,y1min,x1max,y1max)
for (x,y) in gornja_sve_koord:
    cv2.putText(frame, '.',(x,y),cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,50)) 

# In[91]:




from scipy import signal


import numpy as np

def detektuj(img1):
    img = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    plt.imshow(img, 'gray')
    edges = cv2.Canny(img, 80, 150) 
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 25,  150)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
def select_roi(image_orig, image_bin):
    sorted_regions = [] # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    koordinate = []
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours: 
        area = cv2.contourArea(contour)
        x,y,w,h = cv2.boundingRect(contour) #koordinate i velicina granicnog pravougaonika
        if area > 0 and h < 600 and h > 12 and w > 8:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            region = image_bin[y:y+h+1,x:x+w+1]
            regions_array.append([resize_region(region), (x,y,w,h)])       
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
            koordinate.append([resize_region(region),x,y])
            img =cv2.subtract(255, resize_region(region))
            plt.imshow(img.reshape(28, 28), cmap='Greys')
            image1=resize_region(region)
            plt.imshow(img.reshape(28, 28), cmap='Greys')
            pred = model.predict(img.reshape(1, 28, 28, 1))
            #print(x," ",y," ",w," ",h)
            
            cv2.putText(image_orig, str(pred.argmax()),(x,y),cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,255))
            global donja_sve_koord
            global gornja_sve_koord
            global konacnaSuma
            global vecUracunateZaPlus
            global vecUracunateZaMinus
            
            potvrdaPlus= True
            potvrdaMinus = True
            
            for slika in vecUracunateZaMinus:
                if len(slika) == len(img):
                    for delic in img:
                        if len(delic) == len(slika[0]):
                            if np.all(slika==img):
                                potvrdaMinus=False
                                break;
            for slika in vecUracunateZaPlus:
                if len(slika) == len(img):
                    for delic in img:
                        if len(delic) == len(slika[0]):
                            if np.all(slika==img):
                                potvrdaPlus=False
                                break;
            if potvrdaPlus or not vecUracunateZaPlus :
                if (x+w,y+h) in donja_sve_koord:
                    #if np.any((x+w,y+h) in prom):
                    print("plus ",pred.argmax())
                    konacnaSuma = konacnaSuma + pred.argmax()
                    vecUracunateZaPlus.append(img)
                    break;
            if potvrdaMinus or not vecUracunateZaMinus:            
                if (x+w,y+h) in gornja_sve_koord:
                   # if np.any((x+w,y+h) in prom1):
                    print("minus ",pred.argmax())
                    konacnaSuma = konacnaSuma - pred.argmax()
                    vecUracunateZaMinus.append(img);
                    break;
            cv2.putText(image_orig, str(konacnaSuma),(100,100),cv2.FONT_HERSHEY_TRIPLEX, 5, (255,0,255))
                  
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]
    
    
    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions, koordinate        

cap = cv2.VideoCapture(nazivVidea)
ret, frame = cap.read()
if ret:
    detektuj(frame);
potvrda = True   
konacnaSuma=0;
vecUracunateZaPlus=[];
vecUracunateZaMinus=[];
while(cap.isOpened()):
#cap.isOpened()
    
    
    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      
    #frame =erode(frame1) # uniste se brojevi
    # dalje se sa frejmom radi kao sa bilo kojom drugom slikom, npr
    frame_gray = gray_scale(frame)
   
    #frame_gray1 = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    
    #frame_gray =dilate(erode(frame_gray1)) # uniste se brojevi
    edges = cv2.Canny(frame_gray, 80, 150)
     
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 25,  150)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 3)
    
    #cv2.imshow('frame',frame)
    #cv2.imshow('edges', edges)
    img, contours, hierarchy = cv2.findContours(frame_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img = edges.copy()
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    #plt.imshow(img)
    
    
    detekcija(frame)
    image_color =  cv2.cvtColor(cv2.imread('IMG.jpg'), cv2.COLOR_BGR2RGB)
    img = 255-(image_bin(gray_scale(image_color))) # na kraju se vrsi invertovanje
    img_bin = erode(dilate(img))
    selected_regions, numbers, koord = select_roi(image_color.copy(), img);
    #print(numbers)
    brojac =0;
    cv2.imshow('selektovani',selected_regions)
    potvrda = False
    #print(konacnaSuma)
    ret, frame = cap.read()
cap.release()
cv2.destroyAllWindows()
print(konacnaSuma)

# In[73]:




