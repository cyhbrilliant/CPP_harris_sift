#include <opencv.hpp>
#include <iostream>
#include "harris.h"
#include "draw_Fearure.h"
#include <time.h>
#include "feature_match.h"
using namespace cv;
using namespace std;

int main()
{
	IplImage *img1=cvLoadImage("test10.jpg");
	IplImage *img2=cvLoadImage("test10.jpg");

	int *featurepoint1_num=(int*)malloc(sizeof(int));
	CvMat *featurePoint1;
	int *featurepoint2_num=(int*)malloc(sizeof(int));
	CvMat *featurePoint2;


	cout<<"Harris检测"<<endl;
	long begin_harris=clock();
	featurePoint1=FindHarris(img1,featurepoint1_num);
	
	featurePoint2=FindHarris(img2,featurepoint2_num);
	long end_harris=clock();
	cout<<end_harris-begin_harris<<"ms"<<endl;

	//=======================================特征点寻找完毕
	cout<<"特征点匹配"<<endl;
	CvMat *FeatureMatch;
	int *MatchNum=(int *)malloc(sizeof(int));
	long begin_feature=clock();
	FeatureMatch=FeatureMatching(img1,img2,featurePoint1,featurePoint2,featurepoint1_num,featurepoint2_num,MatchNum,0.4);
	long end_feature=clock();
	cout<<end_feature-begin_feature<<"ms"<<endl;

	//======================================特征点匹配完毕

	drawfeature(img1,img2,featurePoint1,featurePoint2,FeatureMatch,MatchNum);
	cout<<"匹配点数目"<<*MatchNum<<endl;

	system("pause");

	return 0;

}