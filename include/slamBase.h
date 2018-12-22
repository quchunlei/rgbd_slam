
#pragma once

#include <fstream>
#include <vector>
#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;


typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

struct CAMERA_INTRINSIC_PARAMETERS
{
  double cx,cy,fx,fy,scale;
};

PointCloud::Ptr image2PointCloud(cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera );


cv::Point3f point2dTo3d(cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera );

struct FRAME
{
  cv::Mat rgb,depth;
  cv::Mat desp;
  vector<cv::KeyPoint> kp;
};

struct RESULT_OF_PNP
{
  cv::Mat rvec, tvec;
  int inliers;
};

//同时提取关键点和描述子
void computeKeyPointsAndDesp( FRAME& frame, string detector, string descriptor);
//输入：帧1，帧2，相机内参
//estimateMotion 计算两帧图像之间的运动
RESULT_OF_PNP estimateMotion(FRAME& fram1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera);
//旋转矢量和位移矢量转变成变换矩阵
Eigen::Isometry3d cvMat2Eigen( cv::Mat& rvec, cv::Mat& tvec );

PointCloud::Ptr joinPointCloud( PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d T, CAMERA_INTRINSIC_PARAMETERS& camera );

// 
class ParameterReader
{
public:
  ParameterReader(string filename = "./src/parameter.txt")
  {
    ifstream fin(filename.c_str());
    
    if(!fin)
    {
       cerr<<"parameter file does not exist."<<endl;
       return;
    }
    
    while(!fin.eof())
    {
      string str;
      getline(fin, str);
      if(str[0]=='#')
      {
	continue;
      }
      
      int pos = str.find("=");
      if(pos == -1)
      {
	continue;
      }
      
      string key = str.substr(0,pos);
      string value = str.substr(pos+1, str.length());
      data[key] = value;
      
      if( !fin.good())
	break;
    }
  }
  
  string getData( string key)
  {
    map<string,string>::iterator it = data.find(key);
    if(it == data.end())
    {
      cerr<<"Parameter name"<<key<<"not found!"<<endl;
      return string("NOT_FOUND");
    }
    
    return it->second;
  }
  
public:
  map<string,string> data;
};