#ifndef feature_match
#define feature_match


#include <highgui.hpp>
#include "his.h"
#include <time.h>
#include "kdtree.h"
float Mod(CvMat *NowMat,int lx,int ly)// l
{
	float xp1_y=*(NowMat->data.db+(ly)*NowMat->step/8+lx+1);
	float xs1_y=*(NowMat->data.db+(ly)*NowMat->step/8+lx-1);
	float x_yp1=*(NowMat->data.db+(ly+1)*NowMat->step/8+lx);
	float x_ys1=*(NowMat->data.db+(ly-1)*NowMat->step/8+lx);

	float pointmod=sqrt(pow(xp1_y-xs1_y,2)+pow(x_yp1-x_ys1,2));

	return pointmod;
}

float gussian_value(float x,float y,float sigma)
{
	float gv=exp(-(x*x+y*y)/(2*sigma*sigma));
	return gv;
}


CvMat *O_dis(CvMat *AllFeature1,CvMat *AllFeature2,int *f1num,int *f2num)
{

	//O_dist存放计算好的欧式距离 行为F1的每个特征点 列为每个F1特征点对于与F2特征点的欧式距离
	CvMat *O_dist=cvCreateMat(*f1num,*f2num,CV_64FC1);

	float O=0;
	float temp=0;

	for (int i=0;i<*f1num;i++)
	{
		for (int j=0;j<*f2num;j++)
		{
			temp=0;
			for (int k=0;k<128;k++)
			{
				double lp1=*(AllFeature1->data.db+k*AllFeature1->step/8+i);
				double lp2=*(AllFeature2->data.db+k*AllFeature2->step/8+j);
				temp+=pow(lp1-lp2,2.0);
			}
			O=sqrt(temp);
			*(O_dist->data.db+i*O_dist->step/8+j)=O;
		}
	}
	return O_dist;
}






CvMat *FeatureMatching(IplImage *img1,IplImage *img2,CvMat *featurePoint1,CvMat *featurePoint2,int *featurePoint1_num,int *featurePoint2_num,int *MatchNum,float ratio)
{
	IplImage *img1_gray=cvCreateImage(cvGetSize(img1),IPL_DEPTH_8U,1);
	IplImage *img2_gray=cvCreateImage(cvGetSize(img2),IPL_DEPTH_8U,1);

	if (img1->nChannels==1)
	{
		cvCopy(img1,img1_gray);
	}
	else
	{
		cvCvtColor(img1,img1_gray,CV_RGB2GRAY);
	}

	if (img2->nChannels==1)
	{
		cvCopy(img2,img2_gray);
	}
	else
	{
		cvCvtColor(img2,img2_gray,CV_RGB2GRAY);
	}
	
	CvMat *mat1=cvCreateMat(img1->height,img1->width,CV_64FC1);
	CvMat *mat2=cvCreateMat(img2->height,img2->width,CV_64FC1);

	cvConvert(img1_gray,mat1);
	cvConvert(img2_gray,mat2);


	CvMat *main_direction1=cvCreateMat(1,10000,CV_64FC1);
	CvMat *main_direction2=cvCreateMat(1,10000,CV_64FC1);

	CvMat *direction1=cvCreateMat(36,*featurePoint1_num,CV_64FC1);
	CvMat *direction2=cvCreateMat(36,*featurePoint2_num,CV_64FC1);

	cvZero(direction1);
	cvZero(direction2);

	//======================================================可选
	//===============================计算邻域

	int directionR=4;

	//===============================计算图1梯度
	for (int i=0;i<*featurePoint1_num;i++)
	{
		for (int m=-directionR;m<=directionR;m++)
		{
			int y=*(featurePoint1->data.db+1*featurePoint1->step/8+i)+m;
			if ((y<=0)||(y>=mat1->rows-1))
			{
				continue;
			}
			for (int n=-directionR;n<=directionR;n++)
			{
				int x=*(featurePoint1->data.db+0*featurePoint1->step/8+i)+n;
				if ((x<=0)||(x>=mat1->cols-1))
				{
					continue;
				}

				float dx=*(mat1->data.db+y*mat1->step/8+x+1)-*(mat1->data.db+y*mat1->step/8+x-1);
				float dy=*(mat1->data.db+(y+1)*mat1->step/8+x)-*(mat1->data.db+(y-1)*mat1->step/8+x);

				float g_dire=fastAtan2(-dy,-dx);

				float direBinf=g_dire/10.0;
				int direBin=cvRound(direBinf);
				if (direBin==36)
				{
					direBin=0;
				}

				float direMod=Mod(mat1,x,y)*gussian_value(n,m,2.0);

				*(direction1->data.db+direBin*direction1->step/8+i)+=direMod;

			}
		}
	}

	//===============================计算图2梯度
	for (int i=0;i<*featurePoint2_num;i++)
	{
		for (int m=-directionR;m<=directionR;m++)
		{
			int y=*(featurePoint2->data.db+1*featurePoint2->step/8+i)+m;
			if ((y<=0)||(y>=mat2->rows-1))
			{
				continue;
			}
			for (int n=-directionR;n<=directionR;n++)
			{
				int x=*(featurePoint2->data.db+0*featurePoint2->step/8+i)+n;
				if ((x<=0)||(x>=mat2->cols-1))
				{
					continue;
				}

				float dx=*(mat2->data.db+y*mat2->step/8+x+1)-*(mat2->data.db+y*mat2->step/8+x-1);
				float dy=*(mat2->data.db+(y+1)*mat2->step/8+x)-*(mat2->data.db+(y-1)*mat2->step/8+x);

				float g_dire=fastAtan2(-dy,-dx);

				float direBinf=g_dire/10.0;
				int direBin=cvRound(direBinf);
				if (direBin==36)
				{
					direBin=0;
				}

				float direMod=Mod(mat2,x,y)*gussian_value(n,m,2.0);

				*(direction2->data.db+direBin*direction2->step/8+i)+=direMod;

			}
		}
	}

	//===============================计算图1主辅方向

	int nowfetnum1=*featurePoint1_num;
	for (int i=0;i<nowfetnum1;i++)
	{
		float bigmod=0;
		int bigbin=0;
		for (int j=0;j<36;j++)
		{
			if (*(direction1->data.db+j*direction1->step/8+i)>bigmod)
			{
				bigmod=*(direction1->data.db+j*direction1->step/8+i);
				bigbin=j;
			}
		}

		*(main_direction1->data.db+i)=bigbin*10.0;
		
		//float limitmod=0.9*bigmod;

		//for (int j=0;j<36;j++)
		//{
		//	if (*(direction1->data.db+j*direction1->step/8+i)>limitmod)
		//	{
		//		*(main_direction1->data.db+*featurePoint1_num)=j*10.0;
		//		*(featurePoint1->data.db+0*featurePoint1->step/8+*featurePoint1_num)=*(featurePoint1->data.db+0*featurePoint1->step/8+i);
		//		*(featurePoint1->data.db+1*featurePoint1->step/8+*featurePoint1_num)=*(featurePoint1->data.db+1*featurePoint1->step/8+i);
		//		(*featurePoint1_num)++;
		//		
		//	}
		//}
	}

	//===============================计算图2主辅方向
	int nowfetnum2=*featurePoint2_num;
	for (int i=0;i<nowfetnum2;i++)
	{
		float bigmod=0;
		int bigbin=0;
		for (int j=0;j<36;j++)
		{
			if (*(direction2->data.db+j*direction2->step/8+i)>bigmod)
			{
				bigmod=*(direction2->data.db+j*direction2->step/8+i);
				bigbin=j;
			}
		}

		*(main_direction2->data.db+i)=bigbin*10.0;

		//float limitmod=0.9*bigmod;

		//for (int j=0;j<36;j++)
		//{
		//	if (*(direction2->data.db+j*direction2->step/8+i)>limitmod)
		//	{
		//		*(main_direction2->data.db+*featurePoint2_num)=j*10.0;
		//		*(featurePoint2->data.db+0*featurePoint2->step/8+*featurePoint2_num)=*(featurePoint2->data.db+0*featurePoint2->step/8+i);
		//		*(featurePoint2->data.db+1*featurePoint2->step/8+*featurePoint2_num)=*(featurePoint2->data.db+1*featurePoint2->step/8+i);
		//		(*featurePoint2_num)++;

		//	}
		//}
	}

	////===============================绘制图1角点方向
	//IplImage *img1_copy=cvCreateImage(cvGetSize(img1),img1->depth,img1->nChannels);
	//cvCopy(img1,img1_copy);
	//for(int i=0;i<*featurePoint1_num;i++)
	//{
	//	

	//	int x=*(featurePoint1->data.db+0*featurePoint1->step/8+i);
	//	int y=*(featurePoint1->data.db+1*featurePoint1->step/8+i);
	//	float sita=*(main_direction1->data.db+i);

	//	cvLine(img1_copy,cvPoint(x,y),cvPoint(x+15*cos(sita*3.14/180.0),y+15*sin(sita*3.14/180.0)),CV_RGB(250,0,0),1);

	//}

	////===============================绘制图2角点方向
	//IplImage *img2_copy=cvCreateImage(cvGetSize(img2),img2->depth,img2->nChannels);
	//cvCopy(img2,img2_copy);
	//for(int i=0;i<*featurePoint2_num;i++)
	//{
	//	

	//	int x=*(featurePoint2->data.db+0*featurePoint2->step/8+i);
	//	int y=*(featurePoint2->data.db+1*featurePoint2->step/8+i);
	//	float sita=*(main_direction2->data.db+i);

	//	cvLine(img2_copy,cvPoint(x,y),cvPoint(x+15*cos(sita*3.14/180.0),y+15*sin(sita*3.14/180.0)),CV_RGB(250,0,0),1);
	//	//cvCircle(img2_copy,cvPoint(x,y),5,CV_RGB(0,255,0),5);

	//}


	//cvNamedWindow("1",0);
	//cvNamedWindow("2",0);
	//cvShowImage("1",img1_copy);
	//cvShowImage("2",img2_copy);

	//cvWaitKey(0);
	//cvReleaseImage(&img1_copy);
	//cvReleaseImage(&img2_copy);
	//


	CvMat *AllFeature1=cvCreateMat(128,10000,CV_64FC1);
	CvMat *AllFeature2=cvCreateMat(128,10000,CV_64FC1);
	//===============================描述图1特征点
	for(int i=0;i<*featurePoint1_num;i++)
	{

		float ft[128];
		double x=*(featurePoint1->data.db+0*featurePoint1->step/8+i);
		double y=*(featurePoint1->data.db+1*featurePoint1->step/8+i);
		Point2f p(x,y);
		float direct=*(main_direction1->data.db+0*main_direction1->step/8+i);

		calcSIFTDescriptor(mat1,p,direct,ft);

		for (int k=0;k<128;k++)
		{
			*(AllFeature1->data.db+k*AllFeature1->step/8+i)=ft[k];
		}
	}

	//===============================描述图2特征点
	for(int i=0;i<*featurePoint2_num;i++)
	{

		float ft[128];
		float x=*(featurePoint2->data.db+0*featurePoint2->step/8+i);
		float y=*(featurePoint2->data.db+1*featurePoint2->step/8+i);
		Point2f p(x,y);
		float direct=*(main_direction2->data.db+0*main_direction2->step/8+i);

		calcSIFTDescriptor(mat2,p,direct,ft);

		for (int k=0;k<128;k++)
		{
			*(AllFeature2->data.db+k*AllFeature2->step/8+i)=ft[k];
		}
	}



	//CvMat *O_distance=O_dis(AllFeature1,AllFeature2,featurePoint1_num,featurePoint2_num);


	CvMat *FeatureMatch=cvCreateMat(2,10000,CV_64FC1);
	(*MatchNum)=0;

	//===============================================================================kdtree
	long kdbegin=clock();
	Kdtree *kdroot=BuildKdtree(AllFeature2,128,*featurePoint2_num);
	FeatureMatch=matchAll(AllFeature1,*featurePoint1_num,kdroot,128,MatchNum);
	long kdend=clock();
	cout<<"KD:   "<<kdend-kdbegin<<"ms"<<endl;

	//===============================================================================

	//long begin_matchratio=clock();
	//for(int i=0;i<*featurePoint1_num;i++)
	//{

	//	float temp1=100000000.0;
	//	float temp1_num=0;
	//	//for(int j=0;j<*featurePoint2_num;j++)
	//	//{
	//	//	if(*(O_distance->data.db+i*O_distance->step/8+j)<temp1)
	//	//	{
	//	//		temp1=*(O_distance->data.db+i*O_distance->step/8+j);
	//	//		temp1_num=j;
	//	//	}
	//	//}

	//	float temp2=100000000.0;
	//	float temp2_num=0;
	//	//for(int j=0;j<*featurePoint2_num;j++)
	//	//{
	//	//	if((*(O_distance->data.db+i*O_distance->step/8+j)<temp2)&&(*(O_distance->data.db+i*O_distance->step/8+j)>temp1))
	//	//	{
	//	//		temp2=*(O_distance->data.db+i*O_distance->step/8+j);
	//	//		temp2_num=j;
	//	//	}
	//	//}

	//	if(*(O_distance->data.db+i*O_distance->step/8+0)<*(O_distance->data.db+i*O_distance->step/8+1))
	//	{
	//		temp1=*(O_distance->data.db+i*O_distance->step/8+0);
	//		temp1_num=0;
	//		temp2=*(O_distance->data.db+i*O_distance->step/8+1);
	//		temp2_num=1;
	//	}
	//	else
	//	{
	//		temp2=*(O_distance->data.db+i*O_distance->step/8+0);
	//		temp2_num=0;
	//		temp1=*(O_distance->data.db+i*O_distance->step/8+1);
	//		temp1_num=1;
	//	}
	//	for(int j=2;j<*featurePoint2_num;j++)
	//	{
	//		if(*(O_distance->data.db+i*O_distance->step/8+j)<temp2)
	//		{
	//			if (*(O_distance->data.db+i*O_distance->step/8+j)<temp1)
	//			{
	//				temp2=temp1;
	//				temp2_num=temp1_num;
	//				temp1=*(O_distance->data.db+i*O_distance->step/8+j);
	//				temp1_num=j;
	//			}
	//			else
	//			{
	//				temp2=*(O_distance->data.db+i*O_distance->step/8+j);
	//				temp2_num=j;
	//			}
	//		}
	//	}



	//	//cout<<temp1_num<<"__"<<temp2_num<<endl;
	//	if ((temp1/temp2)<ratio)
	//	{
	//		*(FeatureMatch->data.db+0*FeatureMatch->step/8+*MatchNum)=i;
	//		*(FeatureMatch->data.db+1*FeatureMatch->step/8+*MatchNum)=temp1_num;
	//		(*MatchNum)++;
	//	}

	//}
	//long end_matchratio=clock();

	//cout<<"ratio匹配时间为:   "<<end_matchratio-begin_matchratio<<"ms"<<endl;

	//cout<<"匹配数目"<<*MatchNum<<endl;



	return FeatureMatch;


}

//void findMinAndMix2(int a[],int n,int* s1,int* s2)
//{
//	int min1,min2;
//	if(a[0]<a[1])
//	{
//		min1=a[0];
//		min2=a[1];
//	}
//	else
//	{
//		min1=a[1];
//		min2=a[0];
//	}
//	for(int i=2;i<n;i++)
//	{
//		if(a[i]<min2)
//		{
//			if (a[i]<min1)
//			{
//				min2=min1;
//				min1=a[i];
//			}
//			else
//			{
//				min2=a[i];
//			}
//		}
//	}
//	*s1=min1;
//	*s2=min2;
//}




#endif