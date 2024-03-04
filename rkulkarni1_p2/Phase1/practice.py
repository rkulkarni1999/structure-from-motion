import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os 

class EpipolarGeometry: 
    def __init__(self,showImages): 
        # Load Images 
        root = os.getcwd()
        imgLeftPath = os.path.join(root,'demo_images\motorcycle\im0.png')
        imgRightPath = os.path.join(root,'demo_images\motorcycle\im1.png')
        self.imgLeft = cv.imread(imgLeftPath,cv.IMREAD_GRAYSCALE) 
        self.imgRight = cv.imread(imgRightPath,cv.IMREAD_GRAYSCALE) 

        if showImages: 
            plt.figure() 
            plt.subplot(121)
            plt.imshow(self.imgLeft)
            plt.subplot(122)
            plt.imshow(self.imgRight)
            plt.show() 

    def drawStereoEpilines(self):  
        # Feature Matching 
        sift = cv.SIFT_create()
        kpLeft,desLeft = sift.detectAndCompute(self.imgLeft,None)
        kpRight,desRight = sift.detectAndCompute(self.imgRight,None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(desLeft,desRight,k=2)
        ptsLeft = []
        ptsRight = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                ptsRight.append(kpRight[m.trainIdx].pt)
                ptsLeft.append(kpLeft[m.queryIdx].pt)

        # Calc fundamental matrix 
        ptsLeft = np.int32(ptsLeft)
        ptsRight = np.int32(ptsRight)
        F,mask = cv.findFundamentalMat(ptsLeft,ptsRight,cv.FM_LMEDS)
        print(f"Fundamental Matrix : {np.round(F, decimals=3)}")

        # Extract points 
        ptsLeft = ptsLeft[mask.ravel()==1]
        ptsRight = ptsRight[mask.ravel()==1]
        step = 10 
        ptsLeft = ptsLeft[::step,:]
        ptsRight = ptsRight[::step,:]

        # Draw epilines on left and right images 
        linesLeft = cv.computeCorrespondEpilines(ptsRight.reshape(-1,1,2),2,F)
        linesLeft = linesLeft.reshape(-1,3)
        imgLeftLines,_ = EpipolarGeometry.drawLines(self.imgLeft,self.imgRight,linesLeft,ptsLeft,ptsRight)
        linesRight = cv.computeCorrespondEpilines(ptsLeft.reshape(-1,1,2),1,F)
        linesRight = linesRight.reshape(-1,3)
        imgRightLines,_ = EpipolarGeometry.drawLines(self.imgRight,self.imgLeft,linesRight,ptsRight,ptsLeft)
        plt.subplot(121)
        plt.imshow(imgLeftLines)
        plt.subplot(122)
        plt.imshow(imgRightLines)
        plt.show()

    @staticmethod
    def drawLines(imgLeft,imgRight,lines,ptsLeft,ptsRight):
        r,c = imgLeft.shape
        imgLeft = cv.cvtColor(imgLeft,cv.COLOR_GRAY2BGR)
        imgRight = cv.cvtColor(imgRight,cv.COLOR_GRAY2BGR)
        for r,pt1,pt2 in zip(lines,ptsLeft,ptsRight):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int,[0,-r[2]/r[1]])
            x1,y1 = map(int,[c,-(r[2]+r[0]*c)/r[1]])
            imgLeft = cv.line(imgLeft,(x0,y0),(x1,y1),color,1)
            imgLeft = cv.circle(imgLeft,tuple(pt1),5,color,-1)
            imgRight = cv.circle(imgRight,tuple(pt2),5,color,-1)
        return imgLeft,imgRight
    
def demoViewPics(): 
    # See pictures 
    eg = EpipolarGeometry(showImages=True)

def demoDrawEpilines(): 
    # Draw epilines 
    eg = EpipolarGeometry(showImages=False)
    eg.drawStereoEpilines() 

if __name__ == '__main__': 
    # demoViewPics() 
    demoDrawEpilines() 