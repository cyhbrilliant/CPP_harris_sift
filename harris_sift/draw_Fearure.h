#ifndef draw_Feature
#define draw_Feature

#include <opencv.hpp>
#include <iostream>
#include <highgui.hpp>
using namespace cv;
using namespace std;

void drawfeature(IplImage *img1,IplImage *img2,CvMat *featrue_point1,CvMat *featrue_point2,CvMat *match,int *MatchNum)
{
	IplImage *img_big=cvCreateImage(cvSize(img1->width+img2->width,max(img1->height,img2->height)),img1->depth,img1->nChannels);
	
	cvSetImageROI(img_big,cvRect(0,0,img1->width,img1->height));
	cvCopy(img1,img_big);
	cvSetImageROI(img_big,cvRect(img1->width,0,img2->width,img2->height));
	cvCopy(img2,img_big);
	cvResetImageROI(img_big);

	for (int i=0;i<*MatchNum;i++)
	{
		int num1=*(match->data.db+0*match->step/8+i);
		int num2=*(match->data.db+1*match->step/8+i);

		int x1=*(featrue_point1->data.db+0*featrue_point1->step/8+num1);
		int y1=*(featrue_point1->data.db+1*featrue_point1->step/8+num1);
		int x2=*(featrue_point2->data.db+0*featrue_point2->step/8+num2);
		int y2=*(featrue_point2->data.db+1*featrue_point2->step/8+num2);
		
		cvLine(img_big,cvPoint(x1,y1),cvPoint(x2+img1->width,y2),CV_RGB(255,0,0),2);
		cvCircle(img_big,cvPoint(x1,y1),5,CV_RGB(0,255,0),3);
		cvCircle(img_big,cvPoint(x2+img1->width,y2),5,CV_RGB(0,0,255),3);
	}

	cvNamedWindow("result",0);
	cvShowImage("result",img_big);
	cvWaitKey(0);


}


void draw_KDfeature(IplImage *img1,IplImage *img2,CvMat *featrue_point1,CvMat *featrue_point2,CvMat *match,int *MatchNum)
{

}





#endif