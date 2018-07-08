#ifndef harris_H
#define harris_H


#include <iostream>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

class harris
{
private:
	cv::Mat  cornerStrength;  //opencv harris函数检测结果，也就是每个像素的角点响应函数值
	cv::Mat cornerTh; //cornerStrength阈值化的结果
	cv::Mat localMax; //局部最大值结果
	int neighbourhood; //邻域窗口大小
	int aperture;//sobel边缘检测窗口大小（sobel获取各像素点x，y方向的灰度导数）
	double k;
	double maxStrength;//角点响应函数最大值
	double threshold;//阈值除去响应小的值
	int nonMaxSize;//这里采用默认的3，就是最大值抑制的邻域窗口大小
	cv::Mat kernel;//最大值抑制的核，这里也就是膨胀用到的核
public:
	harris():neighbourhood(3),aperture(3),k(0.01),maxStrength(0.0),threshold(0.01),nonMaxSize(3)
	{
	};

	void setLocalMaxWindowsize(int nonMaxSize)
	{
		this->nonMaxSize = nonMaxSize;
	};

	//计算角点响应函数以及非最大值抑制
	void detect(const cv::Mat &image)
	{
		//opencv自带的角点响应函数计算函数
		cv::cornerHarris (image,cornerStrength,neighbourhood,aperture,k);
		double minStrength;
		//计算最大最小响应值
		cv::minMaxLoc (cornerStrength,&minStrength,&maxStrength);

		cv::Mat dilated;
		//默认3*3核膨胀，膨胀之后，除了局部最大值点和原来相同，其它非局部最大值点被
		//3*3邻域内的最大值点取代
		cv::dilate (cornerStrength,dilated,cv::Mat());
		//与原图相比，只剩下和原图值相同的点，这些点都是局部最大值点，保存到localMax
		cv::compare(cornerStrength,dilated,localMax,cv::CMP_EQ);
	}

	//获取角点图
	cv::Mat getCornerMap(double qualityLevel) 
	{
		cv::Mat cornerMap;
		// 根据角点响应最大值计算阈值
		threshold= qualityLevel*maxStrength;
		cv::threshold(cornerStrength,cornerTh,
			threshold,255,cv::THRESH_BINARY);
		// 转为8-bit图
		cornerTh.convertTo(cornerMap,CV_8U);
		// 和局部最大值图与，剩下角点局部最大值图，即：完成非最大值抑制
		cv::bitwise_and(cornerMap,localMax,cornerMap);
		return cornerMap;
	}

	void getCorners(std::vector<cv::Point> &points,double qualityLevel) 
	{
			//获取角点图
			cv::Mat cornerMap= getCornerMap(qualityLevel);
			// 获取角点
			getCorners(points, cornerMap);
	}

	// 遍历全图，获得角点
	void getCorners(std::vector<cv::Point> &points,
		const cv::Mat& cornerMap) 
	{

			for( int y = 0; y < cornerMap.rows; y++ ) 
			{
				const uchar* rowPtr = cornerMap.ptr<uchar>(y);
				for( int x = 0; x < cornerMap.cols; x++ ) 
				{
					// 非零点就是角点
					if (rowPtr[x]) 
					{
						points.push_back(cv::Point(x,y));
					}
				}
			}
	}

	//用圈圈标记角点
	void drawOnImage(cv::Mat &image,const std::vector<cv::Point> &points,cv::Scalar color= cv::Scalar(255,255,255),int radius=3, int thickness=2) 
	{
			std::vector<cv::Point>::const_iterator it=points.begin();
			while (it!=points.end()) 
			{
				// 角点处画圈
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