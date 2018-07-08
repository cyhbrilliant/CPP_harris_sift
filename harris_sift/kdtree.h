#ifndef kdtree
#define kdtree

#include <iostream>
#include <opencv.hpp>
using namespace cv;
using namespace std;

class Kdtree
{
public:
	Kdtree *left;
	Kdtree *right;
	Kdtree *parent;
	int splitV;
	double *data;//=========建立树时需要动态分配内存
	int YNnode;//==========1是节点  0是叶子
	int featureSN;
};



void init_kdnode(Kdtree *kdnode,int V,int splitV)
{
	kdnode->YNnode=1;
	kdnode->data=(double*)malloc(sizeof(double)*V);
	kdnode->left=NULL;
	kdnode->right=NULL;
	kdnode->parent=NULL;
	kdnode->splitV=splitV;
}

int variance(CvMat *AllFeature,int num,int V)//===============返回方差最大的split域(计算到end)
{
	//float *vari=(float*)malloc(sizeof(float)*V);
	//float *average=(float*)malloc(sizeof(float)*V);
	//int num=0;
	int vari_max=0;
	int maxnum=0;
	for (int i=0;i<V;i++)
	{
		float vari_temp=0;
		float average_temp=0;
		for (int j=0;j<num;j++)
		{
			average_temp+=*(AllFeature->data.db+i*AllFeature->step/8+j);
		}
		average_temp=average_temp/(float)num;
		for (int j=0;j<num;j++)
		{
			float data=*(AllFeature->data.db+i*AllFeature->step/8+j);
			vari_temp+=(data-average_temp)*(data-average_temp);
		}
		//*(vari+i)=vari_temp;
		if (vari_temp>vari_max)
		{
			vari_max=vari_temp;
			maxnum=i;
		}
	}

	return maxnum;//======split
}

int FindMiddle(CvMat *lrdata,int num,int splitx,int V)
{
	CvMat *AllFeature=cvCreateMat(lrdata->rows,lrdata->cols,CV_64FC1);
	AllFeature=cvCloneMat(lrdata);

	int middlenum=0;

	int *a=(int *)malloc(sizeof(int)*num);
	for (int i=0;i<num;i++)
	{
		a[i]=i;
	}

	for (int i=0;i<num;i++)
	{
		int max=*(AllFeature->data.db+splitx*AllFeature->step/8+i);
		int k=i;

		for (int j=i+1;j<num;j++)
		{
			if (*(AllFeature->data.db+splitx*AllFeature->step/8+j)>max)
			{
				max=*(AllFeature->data.db+splitx*AllFeature->step/8+j);
				k=j;
			}
		}

		float temp=*(AllFeature->data.db+splitx*AllFeature->step/8+k);
		*(AllFeature->data.db+splitx*AllFeature->step/8+k)=*(AllFeature->data.db+splitx*AllFeature->step/8+i);
		*(AllFeature->data.db+splitx*AllFeature->step/8+i)=temp;

		int t;
		t=a[k];
		a[k]=a[i];
		a[i]=t;
	}

	middlenum=a[(num-1)/2];
	return middlenum;

}


void expandtree(Kdtree *parent,CvMat *lrdata,int V,int num,int LR,int *SN)
{
	if (num==0)
	{
		if (LR==1)
		{
			parent->YNnode=4;
			parent->left=NULL;
		}
		if (LR==2)
		{
			parent->YNnode=3;
			parent->right=NULL;
		}
		return ;
	}
	if (num==1)
	{
		Kdtree *kdnode=(Kdtree*)malloc(sizeof(Kdtree));
		kdnode->parent=parent;
		kdnode->left=NULL;
		kdnode->right=NULL;
		kdnode->YNnode=0;
		kdnode->splitV=-1;
		kdnode->data=(double*)malloc(sizeof(double)*V);
		for (int i=0;i<V;i++)
		{
			kdnode->data[i]=*(lrdata->data.db+i*lrdata->step/8);
		}
		kdnode->featureSN=SN[0];
		return ;
	}



	Kdtree *kdnode;
	kdnode=(Kdtree*)malloc(sizeof(Kdtree));

	int splitx=variance(lrdata,num,V);
	int middlenum=FindMiddle(lrdata,num,splitx,V);

	init_kdnode(kdnode,V,splitx);
	kdnode->parent=parent;
	if (LR==1)
	{
		parent->left=kdnode;
	}
	if (LR==2)
	{
		parent->right=kdnode;
	}
	for(int i=0;i<V;i++)
	{
		kdnode->data[i]=*(lrdata->data.db+i*lrdata->step/8+middlenum);
	}

	kdnode->featureSN=SN[middlenum];

	int *SNL=(int *)malloc(sizeof(int)*1000);
	int *SNR=(int *)malloc(sizeof(int)*1000);


	CvMat *leftdata=cvCreateMat(128,1000,CV_64FC1);
	CvMat *rightdata=cvCreateMat(128,1000,CV_64FC1);
	int leftnum=0;
	int rightnum=0;
	for (int i=0;i<num;i++)
	{
		if (i!=middlenum)
		{
			if (*(lrdata->data.db+splitx*lrdata->step/8+i)<=kdnode->data[splitx])
			{
				for (int j=0;j<V;j++)
				{
					*(leftdata->data.db+j*leftdata->step/8+leftnum)=*(lrdata->data.db+j*lrdata->step/8+i);
					SNL[leftnum]=SN[i];
				}
				leftnum++;
			}
			if (*(lrdata->data.db+splitx*lrdata->step/8+i)>kdnode->data[splitx])
			{
				for (int j=0;j<V;j++)
				{
					*(rightdata->data.db+j*rightdata->step/8+rightnum)=*(lrdata->data.db+j*lrdata->step/8+i);
					SNR[rightnum]=SN[i];
				}
				rightnum++;
			}
		}
	}
	/*cout<<"bbb"<<leftnum<<endl;
	cout<<"aaa"<<rightnum<<endl;*/
	//FILE *file;
	//file=fopen("D:\\split.txt","a+");
	////for (int i=0;i<V;i++)
	////{
	////	fprintf(file,"%d\n",splitx);
	////}
	//////fprintf(file,"\n");
	//fprintf(file,"%d    %d\n",leftnum,rightnum);
	//fclose(file);
	expandtree(kdnode,leftdata,V,leftnum,1,SNL);
	expandtree(kdnode,rightdata,V,rightnum,2,SNR);
	cvReleaseMat(&leftdata);
	cvReleaseMat(&rightdata);
	
}

Kdtree * BuildKdtree(CvMat *AllFeature,int V,int num)//===========AllFeature : cvMat(V,num)    V : 维数   num : 特征点数
{
	Kdtree *kdnode;
	CvMat *lrdata=cvCreateMat(AllFeature->rows,AllFeature->cols,CV_64FC1);
	lrdata=cvCloneMat(AllFeature);


	int splitx=variance(lrdata,num,V);
	int middlenum=FindMiddle(lrdata,num,splitx,V);

	kdnode=(Kdtree*)malloc(sizeof(Kdtree));
	init_kdnode(kdnode,V,splitx);

	for(int i=0;i<V;i++)
	{
		kdnode->data[i]=*(AllFeature->data.db+i*AllFeature->step/8+middlenum);
	}
	kdnode->featureSN=middlenum;

	int *SNL=(int *)malloc(sizeof(int)*1000);
	int *SNR=(int *)malloc(sizeof(int)*1000);

	CvMat *leftdata=cvCreateMat(128,1000,CV_64FC1);
	CvMat *rightdata=cvCreateMat(128,1000,CV_64FC1);
	int leftnum=0;
	int rightnum=0;


	for (int i=0;i<num;i++)
	{
		if (i!=middlenum)
		{
			if (*(lrdata->data.db+splitx*lrdata->step/8+i)<=kdnode->data[splitx])
			{
				for (int j=0;j<V;j++)
				{
					*(leftdata->data.db+j*leftdata->step/8+leftnum)=*(lrdata->data.db+j*lrdata->step/8+i);
					SNL[leftnum]=i;
				}
				leftnum++;
			}
			if (*(lrdata->data.db+splitx*lrdata->step/8+i)>kdnode->data[splitx])
			{
				for (int j=0;j<V;j++)
				{
					//cout<<"test  "<<rightnum<<endl;
					*(rightdata->data.db+j*rightdata->step/8+rightnum)=*(lrdata->data.db+j*lrdata->step/8+i);
					SNR[rightnum]=i;
				}
				rightnum++;
			}
		}
	}

	expandtree(kdnode,leftdata,V,leftnum,1,SNL);
	expandtree(kdnode,rightdata,V,rightnum,2,SNR);
	cvReleaseMat(&leftdata);
	cvReleaseMat(&rightdata);
	return kdnode;
}

class prelist
{
public:
	prelist *next;
	int disance;
	Kdtree *kdnode;
};

prelist* PushNode(prelist *prehead,Kdtree *kdnode,int distance)//================距离从小到大排序完成
{
	prelist *px=prehead;
	//while (px)
	//{
	//	cout<<px->disance<<" -> ";
	//	px=px->next;
	//}
	//cout<<endl;
	//cout<<"pushnode     "<<kdnode->splitV<<endl;

	prelist *head=prehead;

	if (head==NULL)//==============没有节点
	{
		prelist *First=(prelist*)malloc(sizeof(prelist));
		First->disance=distance;
		First->kdnode=kdnode;
		First->next=NULL;
		return First;
	}


	prelist *prenode=(prelist*)malloc(sizeof(prelist));
	prenode->disance=distance;
	prenode->kdnode=kdnode;


	if (head->next==NULL)//====================只有一个节点
	{
		if (distance<head->disance)
		{
			prenode->next=head;
			return prenode;
		}
		if (distance>=head->disance)
		{
			head->next=prenode;
			prenode->next=NULL;
			return head;
		}
	}
	if (distance<head->disance)
	{
		prenode->next=head;
		return prenode;
	}
	while (head)
	{
		if (head->next)
		{
			/*cout<<" test "<<head->disance<<endl;
			cout<<" test "<<head->next->disance<<endl;*/
			if ((distance>=head->disance)&&(distance>=head->next->disance))
			{
				head=head->next;
				continue;
			}
			if ((distance>=head->disance)&&(distance<=head->next->disance))
			{
				prenode->next=head->next;
				head->next=prenode;
				return prehead;
			}
		}
		else
		{
			head->next=prenode;
			prenode->next=NULL;
			return prehead;
		}


	}
}




class neibor
{
public:
	Kdtree *kdnode;
	float ods;
};

int prenum(prelist *pre)
{
	int a=0;
	while(pre)
	{
		pre=pre->next;
		a++;
	}
	return a;
}

void init_neibor(neibor *nei,Kdtree *kdnode,float feature[],int V)
{
	float temp=0;
	for (int i=0;i<V;i++)
	{
		temp+=(feature[i]-kdnode->data[i])*(feature[i]-kdnode->data[i]);
	}
	float ods=sqrt(temp);
	nei->ods=ods;
	nei->kdnode=kdnode;
}

int match(float feature[],Kdtree *kdroot,int V)//=========================返回最邻近序号
{
	neibor *nei[2];
	nei[0]=(neibor*)malloc(sizeof(neibor));
	nei[1]=(neibor*)malloc(sizeof(neibor));
	init_neibor(nei[0],kdroot,feature,V);
	init_neibor(nei[1],kdroot,feature,V);


	prelist *pre=NULL;
	pre=PushNode(pre,kdroot,100000);

	int l=0;
	int r=0;

	int runtime=0;

	Kdtree *kdnode;
	while (pre)
	{
		kdnode=pre->kdnode;
		pre=pre->next;
		//cout<<"pre change"<<endl;
		while (kdnode)
		{
			runtime++;
			//cout<<"fafa"<<kdnode->YNnode<<endl;
			float temp=0;
			for (int i=0;i<V;i++)
			{
				temp+=(feature[i]-kdnode->data[i])*(feature[i]-kdnode->data[i]);
			}

			float ods=sqrt(temp);
			//float ods=temp;
			//cout<<"ods "<<ods<<endl;

			if (ods<nei[1]->ods)
			{
				if (ods<nei[0]->ods)
				{
					nei[1]->kdnode=nei[0]->kdnode;
					nei[1]->ods=nei[0]->ods;
					nei[0]->kdnode=kdnode;
					nei[0]->ods=ods;
				}
				else
				{
					nei[1]->kdnode=kdnode;
					nei[1]->ods=ods;
				}
			}

			if (!(kdnode->left)&&!(kdnode->right))
			{
				//cout<<"0000"<<endl;
				break;
			}
			else if (feature[kdnode->splitV]<=kdnode->data[kdnode->splitV])
			{
				float distance=abs(feature[kdnode->splitV]-kdnode->data[kdnode->splitV]);
				//if ((kdnode->YNnode==4)||(kdnode->YNnode==1))
				if (kdnode->right)
				{
					pre=PushNode(pre,kdnode->right,distance);
				}
				//if ((kdnode->YNnode==3)||(kdnode->YNnode==1))
				if(kdnode->left)
				{
					kdnode=kdnode->left;
					continue;
				}
				
				break;
			}
			else if (feature[kdnode->splitV]>kdnode->data[kdnode->splitV])
			{
				float distance=abs(feature[kdnode->splitV]-kdnode->data[kdnode->splitV]);
				//if ((kdnode->YNnode==3)||(kdnode->YNnode==1))
				if(kdnode->left)
				{
					pre=PushNode(pre,kdnode->left,distance);
				}
				//if ((kdnode->YNnode==4)||(kdnode->YNnode==1))
				if(kdnode->right)
				{
					kdnode=kdnode->right;
					continue;
				}
				break;
				
			}
			cout<<"error"<<endl;
		}
		if (runtime>100)
		{
			break;
		}
	}

	float ratio=0.6;
	if (nei[0]->ods/nei[1]->ods<ratio)
	{
		/*CvMat *neiFS=cvCreateMat(1,V,CV_64FC1);
		for(int j=0;j<V;j++)
		{
			*(neiFS->data.db+0*neiFS->step/8+j)=nei[0]->kdnode->data[j];
		}*/
		//FILE* file=fopen("D:\\KDfeature.txt","a+");
		//fprintf(file,"特征点  %d\n",nei[0]->kdnode->featureSN);
		//for(int j=0;j<V;j++)
		//{
		//	fprintf(file,"%10lf",nei[0]->kdnode->data[j]);
		//	if ((j%8+1)==0)
		//	{
		//		fprintf(file,"\n");
		//	}
		//}
		//fprintf(file,"\n");
		//fclose(file);
		return nei[0]->kdnode->featureSN;
	}
	else
	{
		 return NULL;
	}

	


}
CvMat * matchAll(CvMat *AllFeature1,int num,Kdtree *kdroot,int V,int *matchnum)
{
	float *feature;
	feature=(float*)malloc(sizeof(float)*V);
	CvMat *featurematch=cvCreateMat(2,10000,CV_64FC1);
	int featurematch_num=0;
	for (int i=0;i<num;i++)
	{
		for (int j=0;j<V;j++)
		{
			feature[j]=*(AllFeature1->data.db+j*AllFeature1->step/8+i);
		}
		int Fmacth=match(feature,kdroot,V);
		if (Fmacth!=NULL)
		{
			*(featurematch->data.db+0*featurematch->step/8+featurematch_num)=i;
			*(featurematch->data.db+1*featurematch->step/8+featurematch_num)=Fmacth;
			featurematch_num++;
		}

	}

	*matchnum=featurematch_num;

	cout<<"数量：   "<<featurematch_num<<endl;
	return featurematch;




}







#endif