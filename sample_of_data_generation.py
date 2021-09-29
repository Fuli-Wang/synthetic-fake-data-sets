import cv2
import os
import random
import numpy as np
import math

#output   Directories where the data set will be saved
PATHDS = r"C:\Users\fuliw\Documents\python\Data_Generation\Data\DATA_ROT_VGA"
PATHDSGT = r"C:\Users\fuliw\Documents\python\Data_Generation\Data\DATA_ROT_VGA_GT"

#Input  Directories of background, fruits and leafs
PATHBG = r"C:\Users\fuliw\Documents\python\Data_Generation\Background\Backgroundr"
PATHFG = "C:/Users/Documents/python/Data_Generation/fruits-360\Training/"
PATHL = r"C:\Users\fuliw\Documents\python\Data_Generation\Background\leafs"

#Creating data set directories
os.mkdir(PATHDS)
os.mkdir(PATHDSGT)

#Listing content of directories
bg_urls = [f for f in os.listdir(PATHBG)]
#fg_urls = [f for f in os.listdir(PATHFG)]
l_urls = [f for f in os.listdir(PATHL)]


#Listing of fruits to be used in the data set
fruits = ["Strawberry"]

#Dimension of the data set
H = 720
W = 1280


def adjust_gamma(image, gamma=1.0):
#Function that modifies the lightning of the given image.

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255

      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

#Counting variables
c = 0
cc = 0

def options_gen(x, y):
#Generate the different positions and number of fruit image that can fit in the background image

    options_org = []

    X_max=int(W/x)-1

    Y_max=int(H/y)-1

    for ii in range(X_max+1):
        for ji in range(Y_max+1):
            options_org.append((ji,ii))

    dX = int((W-x)/(X_max))
    dY = int((H-y)/(Y_max))

    return options_org, dX, dY, X_max, Y_max

fg_urls = []

#Iterate for each fruit selected above.
for fruit in fruits:

    #list content of the specific fruit directory
    fg_urls = [f for f in os.listdir(PATHFG + fruit)]

    #Using an example of the directory to generate the positions
    example = cv2.imread(PATHFG + fruit + "/" + fg_urls[0],1)
    options_org, dX, dY, X_max, Y_max = options_gen(example.shape[1], example.shape[0])

    #Inizilise counter
    c = 0


    for n in range(30):

        #Preparing the fruits and backgrounds
        random.shuffle(fg_urls)

        bg_urls = []
        bg_urls = [f for f in os.listdir(PATHBG)]

        for xx in range(30):

            #The number of crops that will be displayed
            r = random.randint(1,X_max*Y_max)

            options = options_org.copy()
            random.shuffle(l_urls)
            fg_coor = []
            fg = []
            lf = []

            #For every crop that will be displayed
            for i in range(r):

                #Read the crop, position
                fg.append(cv2.imread(PATHFG + fruit + "/" + fg_urls[i],1))
                r1 = random.choice(range(len(options)))
                fg_coor.append(options[r1])
                options.pop(r1)
                rand_num =  random.randint(1,3)

                #Read leaf and rotate it to create randomness
                leaf = l_urls[(i+1)%len(l_urls)]
                leaf_img = cv2.imread(PATHL + leaf,0)

                for rr in range(rand_num):
                    leaf_img = cv2.rotate(leaf_img, cv2.ROTATE_90_CLOCKWISE)

                leaf_img = cv2.resize(leaf_img, (example.shape[1], example.shape[0]))

                lf.append(leaf_img)


            #Generate gamma's for each crop and background
            gamma_list = []
            for num in range(r+1): gamma_list.append(random.uniform(0.5,2))

            #Iterate two times, one for the Ground Truth and the other for the input
            for ds in [True, False]:

                #Inisilize images
                out = np.zeros((H,W,3), np.uint8)
                white = np.zeros((H,W,3), np.uint8)
                white[:,:] = (255,255,255)
                mask = np.zeros((H,W,1), np.uint8)

                #For each crop, read coordinate, modify brightness and add it to the final image
                for i in range(r):


                    (r1,r2) = fg_coor[i]
                    image = adjust_gamma(fg[i], gamma = gamma_list[i])


                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
                    ret, mask_img_og = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)

                    mask_img_og = cv2.bitwise_and(lf[i], mask_img_og)

                    image = cv2.bitwise_and(image, image, mask = mask_img_og)

                    missing_Y = (H - example.shape[0] - (r1*dY+(Y_max-r1)*dY))/2 if H != example.shape[0] else 0
                    missing_X = (W - example.shape[1] - (r2*dX+(X_max-r2)*dX))/2 if W != example.shape[1] else 0

                    image = cv2.copyMakeBorder(image, r1*dY + math.floor(missing_Y),(Y_max-r1)*dY + math.ceil(missing_Y),r2*dX+ math.floor(missing_X),(X_max-r2)*dX + math.ceil(missing_X), cv2.BORDER_CONSTANT, value = [0,0,0])

                    out = cv2.add(out, image)


                # Create overall foreground and background masks
                gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY )
                ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

                mask_inv = cv2.bitwise_not(thresh)

                bg = random.choice(bg_urls)
                #print(len(bg_urls))
                bg_urls.remove(bg)
                bg = cv2.imread(PATHBG + "/" +bg,1)
                #bg = cv2.imread(PATHBG + y,1)
                bg = cv2.resize(bg, (W,H))
                bg = cv2.bitwise_and(bg, bg, mask = mask_inv)
                white = cv2.bitwise_and(white, white, mask = mask_inv)

                bg = adjust_gamma(bg, gamma = gamma_list[-1])


                # Add the foreground to a white background or selected background
                if ds:
                    out = cv2.add(bg, out)
                else:
                    out = cv2.add(white, out)


                #Display or create the image instance
                if ds:
                    out = cv2.resize(out, (640,480))
                    cv2.imwrite(PATHDS + fruit + "_" + str(c) + ".jpg", out)
                    #cv2.imshow("img", out)
                else:
                    out = cv2.resize(out, (640,480))
                    cv2.imwrite(PATHDSGT + fruit + "_" + str(c) + ".jpg", out)
                    #cv2.imshow("img", out)
                #cv2.waitKey(0)
            c += 1
    cc += 1
    print(cc)
