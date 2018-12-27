#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define T 64

using namespace cv;
using namespace std;

__global__ void Inversion(unsigned char* image, unsigned char* image_inv, int size) {

	int pixel = blockIdx.x*blockDim.x+threadIdx.x;

	unsigned char mcolor = '255';
	if ( pixel < size)
	{
		image_inv[pixel] = mcolor-image[pixel];
	}  
}


int main(int argc, char *argv[])
{
   //scan the filename of image
	string imgfile;
	cout << "Input your image file : ";
	getline (cin, imgfile);

	Mat img = imread(imgfile,IMREAD_COLOR);
	Size imgsize = img.size();
	int width = imgsize.width;
	int height = imgsize.height;
	Mat img_invert(height,width,CV_8UC3,Scalar(0,0,0));

	unsigned char* charImg = img.data;
	unsigned char* newImg = img_invert.data;

	int uCharSize = height*width*3*sizeof(unsigned char);

	unsigned char *devImg,*devInv;

	int vecSize = height*width*3;
	int blocks = (vecSize+T-1)/T;

	cudaMalloc((void**) &devImg, uCharSize);
	cudaMalloc((void**) &devInv, uCharSize);

	cudaMemcpy(devImg,charImg,uCharSize,cudaMemcpyHostToDevice);
	cudaMemcpy(devInv,newImg,uCharSize,cudaMemcpyHostToDevice);

	Inversion<<<blocks,T>>>  (devImg,devInv,vecSize);

	cudaMemcpy(charImg,devImg,uCharSize,cudaMemcpyDeviceToHost);
	cudaMemcpy(newImg,devInv,uCharSize,cudaMemcpyDeviceToHost);

	cudaFree(devImg);
	cudaFree(devInv);
   
	Mat output = Mat(height,width,CV_8UC3, newImg);

	imshow("Your Image",img);
	imshow("Inverted Image",output);
	imwrite("output.jpg",output);
 


	cvWaitKey(0);
}
