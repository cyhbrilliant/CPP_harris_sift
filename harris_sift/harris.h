#ifndef harris_H
#define harris_H


#include <iostream>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

class harris
{
private:
	cv::Mat  cornerStrength;  //opencv harris�����������Ҳ����ÿ�����صĽǵ���Ӧ����ֵ
	cv::Mat cornerTh; //cornerStrength��ֵ���Ľ��
	cv::Mat localMax; //�ֲ����ֵ���
	int neighbourhood; //���򴰿ڴ�С
	int aperture;//sobel��Ե��ⴰ�ڴ�С��sobel��ȡ�����ص�x��y����ĻҶȵ�����
	double k;
	double maxStrength;//�ǵ���Ӧ�������ֵ
	double threshold;//��ֵ��ȥ��ӦС��ֵ
	int nonMaxSize;//�������Ĭ�ϵ�3���������ֵ���Ƶ����򴰿ڴ�С
	cv::Mat kernel;//���ֵ���Ƶĺˣ�����Ҳ���������õ��ĺ�
public:
	harris():neighbourhood(3),aperture(3),k(0.01),maxStrength(0.0),threshold(0.01),nonMaxSize(3)
	{
	};

	void setLocalMaxWindowsize(int nonMaxSize)
	{
		this->nonMaxSize = nonMaxSize;
	};

	//����ǵ���Ӧ�����Լ������ֵ����
	void detect(const cv::Mat &image)
	{
		//opencv�Դ��Ľǵ���Ӧ�������㺯��
		cv::cornerHarris (image,cornerStrength,neighbourhood,aperture,k);
		double minStrength;
		//���������С��Ӧֵ
		cv::minMaxLoc (cornerStrength,&minStrength,&maxStrength);

		cv::Mat dilated;
		//Ĭ��3*3�����ͣ�����֮�󣬳��˾ֲ����ֵ���ԭ����ͬ�������Ǿֲ����ֵ�㱻
		//3*3�����ڵ����ֵ��ȡ��
		cv::dilate (cornerStrength,dilated,cv::Mat());
		//��ԭͼ��ȣ�ֻʣ�º�ԭͼֵ��ͬ�ĵ㣬��Щ�㶼�Ǿֲ����ֵ�㣬���浽localMax
		cv::compare(cornerStrength,dilated,localMax,cv::CMP_EQ);
	}

	//��ȡ�ǵ�ͼ
	cv::Mat getCornerMap(double qualityLevel) 
	{
		cv::Mat cornerMap;
		// ���ݽǵ���Ӧ���ֵ������ֵ
		threshold= qualityLevel*maxStrength;
		cv::threshold(cornerStrength,cornerTh,
			threshold,255,cv::THRESH_BINARY);
		// תΪ8-bitͼ
		cornerTh.convertTo(cornerMap,CV_8U);
		// �;ֲ����ֵͼ�룬ʣ�½ǵ�ֲ����ֵͼ��������ɷ����ֵ����
		cv::bitwise_and(cornerMap,localMax,cornerMap);
		return cornerMap;
	}

	void getCorners(std::vector<cv::Point> &points,double qualityLevel) 
	{
			//��ȡ�ǵ�ͼ
			cv::Mat cornerMap= getCornerMap(qualityLevel);
			// ��ȡ�ǵ�
			getCorners(points, cornerMap);
	}

	// ����ȫͼ����ýǵ�
	void getCorners(std::vector<cv::Point> &points,
		const cv::Mat& cornerMap) 
	{

			for( int y = 0; y < cornerMap.rows; y++ ) 
			{
				const uchar* rowPtr = cornerMap.ptr<uchar>(y);
				for( int x = 0; x < cornerMap.cols; x++ ) 
				{
					// �������ǽǵ�
					if (rowPtr[x]) 
					{
						points.push_back(cv::Point(x,y));
					}
				}
			}
	}

	//��ȦȦ��ǽǵ�
	void drawOnImage(cv::Mat &image,const std::vector<cv::Point> &points,cv::Scalar color= cv::Scalar(255,255,255),int radius=3, int thickness=2) 
	{
			std::vector<cv::Point>::const_iterator it=points.begin();
			while (it!=points.end()) 
			{
				// �ǵ㴦��Ȧ
				cv::circle(image,*it,radius,color,thickness);
				++it;
			}
	}

};


//	x1 x2
//	y1 y2
CvMat *FindHarris(IplImage *src_img,int *cornerNUM)
{
	IplImage *img=cvCreateImage(cvGetSize(src_img),IPL_DEPTH_8U,1);
	if (src_img->nChannels==1)
	{
		cvCopy(src_img,img);
	}
	else
	{
		cvCvtColor(src_img,img,CV_RGB2GRAY);
	}

	Mat image=cvarrToMat(img,true);
	harris Harris;
	Harris.detect(image);

	vector<Point> pts;
	Harris.getCorners(pts,0.01);
	*cornerNUM=pts.size();
	
	CvMat *p_arris1=cvCreateMat(2,10000,CV_64FC1);
	for (int i=0;i<pts.size();i++)
	{
		*(p_arris1->data.db+0*p_arris1->step/8+i)=pts[i].x;
		*(p_arris1->data.db+1*p_arris1->step/8+i)=pts[i].y;
		
	}

	return p_arris1;
	
}





#endif