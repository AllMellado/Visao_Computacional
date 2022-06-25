#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <math.h>


using namespace cv;
using namespace std;


int fillMatrix(float**&m, int tam, int type );
void lowAndHighPixels(Mat src, int &ph, int &pl);
Mat equalize(Mat src);
Mat filterMedium(Mat src);
Mat filterMediumColor(Mat src);
Mat filterBorder(Mat src, float w);
Mat filterCanny(Mat src);
Mat filterLaplace(Mat src);
Mat filterLaplaceGauss(Mat src);
Mat filterRoberts(Mat src);
Mat filterMedian(Mat src);
Mat filterMedianColor(Mat src);
Mat filterGauss(Mat src);
Mat filterGaussColor(Mat src);
Mat cheatSobel(Mat src);
void SelectionSort(int vetor[], int tam);

int main( int argc, char** argv )
{	String fileName = "birds";
	String ext = ".jpg";

	Mat img = imread("./"+fileName+"/"+fileName+ext);
	if(!img.data){
		cout << "Problem loading image "<< endl;
	}
	//imshow("s",img);

	Mat img2, img3;
	if( img.channels() == 3 ){

		img2 = filterGaussColor(img);
		imwrite("./"+fileName+"/"+fileName+"ColorGauss"+ext,img2);

		img2 = filterMedianColor(img);
		imwrite("./"+fileName+"/"+fileName+"ColorMedian"+ext,img2);

		img2 = filterMediumColor(img);
		imwrite("./"+fileName+"/"+fileName+"ColorMedium"+ext,img2);

		cvtColor(img, img,COLOR_BGR2GRAY);
	

	}

	//resize(img, img, Size(1920/4,1280/4));

	img2 = equalize(img);
	imwrite("./"+fileName+"/"+fileName+"0Equilize"+ext,img2);

	img2 = filterMedium(img);
	imwrite("./"+fileName+"/"+fileName+"1Medium"+ext,img2);

	img2 = filterMedian(img);
	imwrite("./"+fileName+"/"+fileName+"2Median"+ext,img2);

	img = filterGauss(img);
	imwrite("./"+fileName+"/"+fileName+"3Gauss"+ext,img);

	img3 = filterBorder(img, 2);
	imwrite("./"+fileName+"/"+fileName+"4Sobel"+ext,img3);

	img3 = filterBorder(img, 1);
	imwrite("./"+fileName+"/"+fileName+"5Prewitt"+ext,img3);

	img3 = filterLaplace(img);
	imwrite("./"+fileName+"/"+fileName+"6Laplace"+ext,img3);

	img3 = filterLaplaceGauss(img);
	imwrite("./"+fileName+"/"+fileName+"7LaplaceGauss"+ext,img3);

	img3 = filterRoberts(img);
	imwrite("./"+fileName+"/"+fileName+"8Roberts"+ext,img3);

	img3 = filterCanny(img);
	imwrite("./"+fileName+"/"+fileName+"9Canny"+ext,img3);
	

    waitKey(50000);

    return 1;
}

int fillMatrix(float**&m, int tam, int type ){

		m = (float**)malloc(tam*sizeof(float*));
		for(int r = 0; r < tam; r++){
			m[r] = (float*)malloc(tam*sizeof(float));
		}

		switch(type){
			case 1:
				for(int r = 0; r < tam; r++){
					for(int i = 0; i < tam; i++){
						m[r][i] = 1;
					}
				}
				tam = tam*tam;
				break;

		}

		return tam;
}

void lowAndHighPixels(Mat src, int &ph, int &pl){
	ph = pl = src.at<uchar>(0,0);

	for(int y = 0; y < src.cols ; y++){
		for(int x = 0; x < src.rows; x++ ){
			if( src.at<uchar>(x,y) < pl )
				pl = src.at<uchar>(x,y);
			if( src.at<uchar>(x,y) > ph )
				ph = src.at<uchar>(x,y);

		}
	}
}

Mat equalize(Mat src){
	float freq[256] = {};
	float prob[256] = {};
	float probC[256] = {};
	int scale = 255;

	Mat dst = src.clone();

	for(int y = 0; y < src.cols ; y++){
		for(int x = 0; x < src.rows; x++ ){
			freq[src.at<uchar>(x,y)]++;
		}
	}

	for(int i = 0; i < 256 ; i++){
		prob[i] = freq[i]/(src.cols*src.rows);
	}

	for(int i = 0; i < 256 ; i++){
			probC[i] = prob[i] + probC[i-1];
	}

	cout << probC[255] << endl;

	for(int i = 0; i < 256 ; i++){
		probC[i] = probC[i]*scale;
	}

	for(int y = 0; y < src.cols ; y++){
		for(int x = 0; x < src.rows; x++ ){
			dst.at<uchar>(x,y) = probC[src.at<uchar>(x,y)];
		}
	}

	return dst;
}

Mat filterMedian(Mat src){
	int k = 5;
	int border = k/2;
	int r = 0;
	int vector[k*k];

	Mat dst;
	copyMakeBorder(src,dst,border,border,border,border,BORDER_REPLICATE);

	for(int y = border; y < src.cols-border; y++){
		for(int x = border; x < src.rows-border; x++ ){
			for(int i = 0; i < k; i++){
				for(int j = 0; j < k; j++){
					vector[r] = src.at<uchar>(x+i-1,y+j-1);
					r++;
				}
			}
			SelectionSort(vector, k*k);
			dst.at<uchar>(x,y) = vector[(k*k)/2];
			r = 0;
		}
	}

	Mat roi(dst, Rect(border, border, src.size().width-k, src.size().height-k));

	return roi;
}

Mat filterMedianColor(Mat src){
	int k = 5;
	int border = k/2;
	int r = 0;
	int vector0[k*k];
	int vector1[k*k];
	int vector2[k*k];


	Mat dst;
	copyMakeBorder(src,dst,border,border,border,border,BORDER_REPLICATE);

	for(int y = border; y < src.cols-border; y++){
		for(int x = border; x < src.rows-border; x++ ){
			for(int i = 0; i < k; i++){
				for(int j = 0; j < k; j++){
					vector0[r] = src.at<Vec3b>(x+i-1,y+j-1)[0];
					vector1[r] = src.at<Vec3b>(x+i-1,y+j-1)[1];
					vector2[r] = src.at<Vec3b>(x+i-1,y+j-1)[2];
					r++;
				}
			}
			SelectionSort(vector0, k*k);
			SelectionSort(vector1, k*k);
			SelectionSort(vector2, k*k);
			dst.at<Vec3b>(x,y)[0] = vector0[(k*k)/2];
			dst.at<Vec3b>(x,y)[1] = vector1[(k*k)/2];
			dst.at<Vec3b>(x,y)[2] = vector2[(k*k)/2];
			r = 0;
		}
	}

	Mat roi(dst, Rect(border, border, src.size().width-k, src.size().height-k));

	return roi;
}

Mat cheatSobel(Mat src){

	Mat grad_x, abs_grad_x;
	Mat grad_y, abs_grad_y;

	Sobel(src, grad_x, CV_16S,1,0,3,1,0,BORDER_DEFAULT);
	convertScaleAbs(grad_x,abs_grad_x);

	Sobel(src, grad_y, CV_16S,0,1,3,1,0,BORDER_DEFAULT);
	convertScaleAbs(grad_y,abs_grad_y);

	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, src);

	return src;
}

Mat filterCanny(Mat src){
	float m1[3][3] = {	{-1,  0, 1},
				   		{-1,  0, 1},
				   		{-1,  0, 1}	};

	float m2[3][3] = {	{-1, -1,-1},
				   		{ 0,  0, 0},
				   		{ 1,  1, 1}	};

	float ang [src.rows][src.cols];

	int k = 3;
	int border = k/2;
	float soma = 0;
	float somaX = 0;
	float somaY = 0;

	Mat dst;
	copyMakeBorder(src,dst,border,border,border,border,BORDER_REPLICATE);

	for(int y = border; y < src.cols-border; y++){
		for(int x = border; x < src.rows-border; x++ ){
			for(int i = 0; i < k; i++){
				for(int j = 0; j < k; j++){
					somaX += src.at<uchar>(x+i-1,y+j-1)*m1[i][j];
					somaY += src.at<uchar>(x+i-1,y+j-1)*m2[i][j];
				}
			}
			soma = sqrt(pow(somaX,2.0) + pow(somaY,2.0));

			ang[x][y] = atan(somaY/somaX)*180/3.14159265;

			if( ang[x][y] > 0 || ang[x][y] < 45 ){
				ang[x][y] = 45;
			}else{
				if( ang[x][y] > 45 || ang[x][y] < 90){
					ang[x][y] = 90;
				}else{
					ang[x][y] = 135;
				}
			}

			dst.at<uchar>(x,y) = soma;
			soma = 0;
			somaX = 0;
			somaY = 0;
		}
	}


	Mat dst2 = dst.clone();
	int a, b, c, d;
	for(int y = border; y < src.cols-border; y++){
		for(int x = border; x < src.rows-border; x++ ){
			switch( (int)(ang[x][y]) ){
				case 0: a = -1; b = 0; c = 1; d = 0; break;

				case 45: a = -1; b = -1; c = 1; d = 1; break;

				case 90: a = 0; b = -1; c = 0; d = 1; break;

				case 135: a = 1; b = -1; c = -1; d = 1; break;

				default: cout << "wtf???" << endl;
			}
			if( x+a >= 0 && x+c >= 0 && y+b >= 0 && x+d >= 0  )
				if( dst.at<uchar>(x,y) <= dst.at<uchar>(x+a,y+b) || dst.at<uchar>(x,y) <= dst.at<uchar>(x+c,y+d) )
					dst2.at<uchar>(x,y) = 0;
		}
	}


	int flag = 0;
	int maxThresh = 60;
	int minThresh = 20;
	for(int y = border; y < src.cols-border; y++){
		for(int x = border; x < src.rows-border; x++ ){
			if( dst2.at<uchar>(x,y) < maxThresh ){
				if( dst2.at<uchar>(x,y) >= minThresh ){
					for(int i = 0; i < k; i++){
						for(int j = 0; j < k; j++ ){
							if(dst2.at<uchar>(x+i,y+j) >= maxThresh){
								flag = 1 ;
							}
						}
					}
					if(flag == 0){
						dst.at<uchar>(x,y) = 0;
					}else{
						dst.at<uchar>(x,y) = 255;
					}
					flag = 0;
				}else{
					dst.at<uchar>(x,y) = 0;
				}
			}else{
				dst.at<uchar>(x,y) = 255;
			}
		}
	}

	Mat roi(dst, Rect(border, border, src.size().width-k, src.size().height-k));

	return roi;
}

Mat filterLaplaceGauss(Mat src){
	float m[5][5] = {	{0, 0,  1,  0, 0},
						{0, 1,  2,  1, 0},
						{1, 2, -16, 2, 1},
						{0, 1,  2,  1, 0},
						{0, 0,  1,  0, 0}};

	int k = 5;
	int border = k/2;
	float soma = 0;

	Mat dst;
	copyMakeBorder(src,dst,border,border,border,border,BORDER_REPLICATE);

	for(int y = border; y < src.cols-border; y++){
		for(int x = border; x < src.rows-border; x++ ){
			for(int i = 0; i < k; i++){
				for(int j = 0; j < k; j++){
					soma += src.at<uchar>(x+i-1,y+j-1)*m[i][j];
				}
			}
			dst.at<uchar>(x,y) = abs(soma) ;
			soma = 0;
		}
	}

	normalize(dst, dst, 3, 0, NORM_MINMAX, CV_32F);
	dst.convertTo(dst, CV_8UC3, 255.0);

	Mat roi(dst, Rect(border, border, src.size().width-k, src.size().height-k));

	return roi;
}

Mat filterLaplace(Mat src){
	float m[3][3] = {	{0, -1, 0},
						{-1, 4, -1},
						{0, -1, 0} };

	int k = 3;
	int border = k/2;
	float soma = 0;

	Mat dst;
	copyMakeBorder(src,dst,border,border,border,border,BORDER_REPLICATE);

	for(int y = border; y < src.cols-border; y++){
		for(int x = border; x < src.rows-border; x++ ){
			for(int i = 0; i < k; i++){
				for(int j = 0; j < k; j++){
					soma  += src.at<uchar>(x+i-1,y+j-1)*m[i][j];
				}
			}
			dst.at<uchar>(x,y) = abs(soma) ;
			soma = 0;
		}
	}
	normalize(dst, dst, 10, 0, NORM_MINMAX, CV_32F);
	dst.convertTo(dst, CV_8UC3, 255.0);

	Mat roi(dst, Rect(border, border, src.size().width-k, src.size().height-k));

	return roi;
}

Mat filterRoberts(Mat src){
	float m1[2][2] = {	{-1, 0},
				   		{ 0, 1} };

	float m2[2][2] = {	{ 0, 1},
				   		{-1, 0} };


	int k = 2;
	float soma = 0;
	float somaX = 0;
	float somaY = 0;

	Mat dst = src.clone();

	for(int y = 0; y < src.cols; y++){
		for(int x = 0; x < src.rows; x++ ){
			for(int i = 0; i < k; i++){
				for(int j = 0; j < k; j++){
					somaX += src.at<uchar>(x+i,y+j)*m1[i][j];
					somaY += src.at<uchar>(x+i,y+j)*m2[i][j];
				}
			}
			soma = sqrt(pow(somaX,2.0) + pow(somaY,2.0));
			dst.at<uchar>(x,y) = soma;
			soma = 0;
			somaX = 0;
			somaY = 0;
		}
	}
	normalize(dst, dst, 3, 0, NORM_MINMAX, CV_32F);
	dst.convertTo(dst, CV_8UC3, 255.0);

	return dst;
}

Mat filterBorder(Mat src, float w){
	float m1[3][3] = {	{-1,  0, 1},
				   		{-w,  0, w},
				   		{-1,  0, 1}	};

	float m2[3][3] = {	{-1, -w,-1},
				   		{ 0,  0, 0},
				   		{ 1,  w, 1}	};


	int k = 3;
	int border = k/2;
	float soma = 0;
	float somaX = 0;
	float somaY = 0;

	Mat dst;
	copyMakeBorder(src,dst,border,border,border,border,BORDER_REPLICATE);

	for(int y = border; y < src.cols-border; y++){
		for(int x = border; x < src.rows-border; x++ ){
			for(int i = 0; i < k; i++){
				for(int j = 0; j < k; j++){
					somaX += src.at<uchar>(x+i-1,y+j-1)*m1[i][j];
					somaY += src.at<uchar>(x+i-1,y+j-1)*m2[i][j];
				}
			}
			soma = sqrt(pow(somaX,2.0) + pow(somaY,2.0));
			dst.at<uchar>(x,y) = soma;
			soma = 0;
			somaX = 0;
			somaY = 0;
		}
	}

	Mat roi(dst, Rect(border, border, src.size().width-k, src.size().height-k));

	return roi;
}

Mat filterMedium(Mat src){

	float m[5][5] = {{1,1,1,1,1},
				     {1,1,1,1,1},
					 {1,1,1,1,1},
					 {1,1,1,1,1},
					 {1,1,1,1,1}};


	int k = 5;
	int border = k/2;
	float soma = 0;

	Mat dst;
	copyMakeBorder(src,dst,border,border,border,border,BORDER_REPLICATE);

	for(int y = border; y < src.cols-border; y++){
		for(int x = border; x < src.rows-border; x++ ){
			for(int i = 0; i < k; i++){
				for(int j = 0; j < k; j++){
					soma = soma + src.at<uchar>(x+i-1,y+j-1)*(m[i][j]/(k*k));
				}
			}
			dst.at<uchar>(x,y) = (int)soma;
			soma = 0;
		}
	}

	Mat roi(dst, Rect(border, border, src.size().width-k, src.size().height-k));

	return roi;
}

Mat filterMediumColor(Mat src){

	float m[5][5] = {{1,1,1,1,1},
				     {1,1,1,1,1},
					 {1,1,1,1,1},
					 {1,1,1,1,1},
					 {1,1,1,1,1}};


	int k = 5;
	int border = k/2;
	float soma0 = 0;
	float soma1 = 0;
	float soma2 = 0;

	Mat dst;
	copyMakeBorder(src,dst,border,border,border,border,BORDER_REPLICATE);

	for(int y = border; y < src.cols-border; y++){
		for(int x = border; x < src.rows-border; x++ ){
			for(int i = 0; i < k; i++){
				for(int j = 0; j < k; j++){
					soma0 += src.at<Vec3b>(x+i-1,y+j-1)[0]*(m[i][j]/(k*k));
					soma1 += src.at<Vec3b>(x+i-1,y+j-1)[1]*(m[i][j]/(k*k));
					soma2 += src.at<Vec3b>(x+i-1,y+j-1)[2]*(m[i][j]/(k*k));
				}
			}
			dst.at<Vec3b>(x,y)[0] = (int)soma0;
			dst.at<Vec3b>(x,y)[1] = (int)soma1;
			dst.at<Vec3b>(x,y)[2] = (int)soma2;

			soma0 = 0;
			soma1 = 0;
			soma2 = 0;
		}
	}

	Mat roi(dst, Rect(border, border, src.size().width-k, src.size().height-k));

	return roi;
}

Mat filterGaussColor(Mat src){
	float m[5][5] = { 	{1,  4,  7,  4, 1},
						{4, 16, 26, 16, 4},
						{7, 26, 41, 26, 7},
						{4, 16, 26, 16, 4},
						{1,  4,  7,  4, 1}};

	int k = 5;
	int border = k/2;
	float soma0 = 0;
	float soma1 = 0;
	float soma2 = 0;

	Mat dst;
	copyMakeBorder(src,dst,border,border,border,border,BORDER_REPLICATE);

	for(int y = border; y < src.cols-border; y++){
		for(int x = border; x < src.rows-border; x++ ){
			for(int i = 0; i < k; i++){
				for(int j = 0; j < k; j++){
					soma0 += src.at<Vec3b>(x+i-1,y+j-1)[0]*(m[i][j]/273);
					soma1 += src.at<Vec3b>(x+i-1,y+j-1)[1]*(m[i][j]/273);
					soma2 += src.at<Vec3b>(x+i-1,y+j-1)[2]*(m[i][j]/273);
				}
			}
			dst.at<Vec3b>(x,y)[0] = (int)soma0;
			dst.at<Vec3b>(x,y)[1] = (int)soma1;
			dst.at<Vec3b>(x,y)[2] = (int)soma2;
			soma0 = 0;
			soma1 = 0;
			soma2 = 0;
		}
	}

	Mat roi(dst, Rect(border, border, src.size().width-k, src.size().height-k));

	return roi;
}

Mat filterGauss(Mat src){
	float m[5][5] = { 	{1,  4,  7,  4, 1},
						{4, 16, 26, 16, 4},
						{7, 26, 41, 26, 7},
						{4, 16, 26, 16, 4},
						{1,  4,  7,  4, 1}};

	int k = 5;
	int border = k/2;
	float soma = 0;

	Mat dst;
	copyMakeBorder(src,dst,border,border,border,border,BORDER_REPLICATE);

	for(int y = border; y < src.cols-border; y++){
		for(int x = border; x < src.rows-border; x++ ){
			for(int i = 0; i < k; i++){
				for(int j = 0; j < k; j++){
					soma += src.at<uchar>(x+i-1,y+j-1)*(m[i][j]/273);
				}
			}
			dst.at<uchar>(x,y) = (int)soma;
			soma = 0;
		}
	}

	Mat roi(dst, Rect(border, border, src.size().width-k, src.size().height-k));

	return roi;
}

void SelectionSort(int vetor[], int tam){
    for (int indice = 0; indice < tam; ++indice) {
        int indiceMenor = indice;
        for (int indiceSeguinte = indice+1; indiceSeguinte < tam; ++indiceSeguinte) {
            if (vetor[indiceSeguinte] < vetor[indiceMenor]) {
                indiceMenor = indiceSeguinte;
            }
        }
        int aux = vetor[indice];
        vetor[indice] = vetor[indiceMenor];
        vetor[indiceMenor] = aux;
    }
}
