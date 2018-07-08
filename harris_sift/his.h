#ifndef his
#define his

#include <iostream>
#include <opencv.hpp>
using namespace cv;
using namespace std;




// SIFT关键点特征描述

void calcSIFTDescriptor( CvMat* img, Point2f ptf, float ori, float* dst )
						   
{
	float SIFT_DESCR_MAG_THR=0.2;
	float SIFT_INT_DESCR_FCTR=512.0;
	int d=4;
	int n=8;
    Point pt(cvRound(ptf.x), cvRound(ptf.y));
	//计算余弦，正弦，CV_PI/180:将角度值转化为幅度值
    float cos_t = cosf(ori*(float)(CV_PI/180));
    float sin_t = sinf(ori*(float)(CV_PI/180));
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f/(d * d * 0.5f);
    float hist_width = 4.0;  
	                                      
	// 计算图像区域半径mσ(d+1)/2*sqrt(2)
	// 1.4142135623730951f 为根号2
    int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
    cos_t /= hist_width;
    sin_t /= hist_width;

    int i, j, k, len = (radius*2+1)*(radius*2+1), histlen = (d+2)*(d+2)*(n+2);
    int rows = img->rows, cols = img->cols;

    AutoBuffer<float> buf(len*6 + histlen);
    float *X = buf, *Y = X + len, *Mag = Y, *Ori = Mag + len, *W = Ori + len;
    float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

	//初始化直方图
    for( i = 0; i < d+2; i++ )
    {
        for( j = 0; j < d+2; j++ )
            for( k = 0; k < n+2; k++ )
                hist[(i*(d+2) + j)*(n+2) + k] = 0.;
    }

	//计算采样区域点坐标旋转
    for( i = -radius, k = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
           
            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d/2 - 0.5f;
            float cbin = c_rot + d/2 - 0.5f;
            int r = pt.y + i, c = pt.x + j;

            if( rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
               r > 0 && r < rows - 1 && c > 0 && c < cols - 1 )
            {
                
				float dx=(float)(*(img->data.db+r*img->step/8+c+1)-*(img->data.db+r*img->step/8+c-1));
				float dy=(float)(*(img->data.db+(r-1)*img->step/8+c)-*(img->data.db+(r+1)*img->step/8+c));
                X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
                W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
                k++;
            }
        }

	len = k;
	for (int i=0;i<len;i++)
	{
		
		Ori[i]=fastAtan2(Y[i], X[i]);
		Mag[i]=sqrt(X[i]*X[i]+Y[i]*Y[i]) ;
		W[i]=exp(W[i]);

	}

	//计算梯度直方图
    for( k = 0; k < len; k++ )
    {
        float rbin = RBin[k], cbin = CBin[k];
        float obin = (Ori[k] - ori)*bins_per_rad;
        float mag = Mag[k]*W[k];

        int r0 = cvFloor( rbin );
        int c0 = cvFloor( cbin );
        int o0 = cvFloor( obin );
        rbin -= r0;
        cbin -= c0;
        obin -= o0;

        //n为SIFT_DESCR_HIST_BINS：8，即将360°分为8个区间
		if( o0 < 0 )
            o0 += n;
        if( o0 >= n )
            o0 -= n;
		

        // histogram update using tri-linear interpolation
		// 双线性插值
        float v_r1 = mag*rbin, v_r0 = mag - v_r1;
        float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
        float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
        float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
        float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
        float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
        float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

        int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;
        hist[idx] += v_rco000;
        hist[idx+1] += v_rco001;
        hist[idx+(n+2)] += v_rco010;
        hist[idx+(n+3)] += v_rco011;
        hist[idx+(d+2)*(n+2)] += v_rco100;
        hist[idx+(d+2)*(n+2)+1] += v_rco101;
        hist[idx+(d+3)*(n+2)] += v_rco110;
        hist[idx+(d+3)*(n+2)+1] += v_rco111;
    }
	// 最后确定直方图，目标方向直方图是圆的
    for( i = 0; i < d; i++ )
        for( j = 0; j < d; j++ )
        {
            int idx = ((i+1)*(d+2) + (j+1))*(n+2);
            hist[idx] += hist[idx+n];
            hist[idx+1] += hist[idx+n+1];
            for( k = 0; k < n; k++ )
                dst[(i*d + j)*n + k] = hist[idx+k];
        }

    float nrm2 = 0;
    len = d*d*n;
    for( k = 0; k < len; k++ )
        nrm2 += dst[k]*dst[k];
    float thr = sqrt(nrm2)*SIFT_DESCR_MAG_THR;
    for( i = 0, nrm2 = 0; i < k; i++ )
    {
        float val = min(dst[i], thr);
        dst[i] = val;
        nrm2 += val*val;
    }
    nrm2 = SIFT_INT_DESCR_FCTR/max(sqrt(nrm2), FLT_EPSILON);
    for( k = 0; k < len; k++ )
    {
        dst[k] = saturate_cast<uchar>(dst[k]*nrm2);
    }
}

#endif