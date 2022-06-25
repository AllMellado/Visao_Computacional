#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/videoio.hpp"

using namespace std;
using namespace cv;

bool try_use_gpu = false;
Stitcher::Mode mode = Stitcher::PANORAMA;
vector<Mat> imgs;

int main(int argc, char* argv[]){
	int frameCount = 0;
	int X = 20; // Quanto maior X --> mais rapido é o algoritmo e menos preciso é o resultado

	VideoCapture video("./video/a.mp4");
	frameCount = video.get(CV_CAP_PROP_FRAME_COUNT)/X;

	cout << "Numero de frames: " << video.get(CV_CAP_PROP_FRAME_COUNT) << endl;
	cout << "Numero de frames usados: " << frameCount << endl;
	cout << "Isso pode demorar um tempo..." << endl;

	video.set(CV_CAP_PROP_FRAME_WIDTH, 840);
	video.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	for(int i = 1; i<=frameCount; i++){
		Mat frame;
		for(int j = 0; j < X; j++)
			video.read(frame);

		imshow("a",frame);
		waitKey(0);

	    if(!frame.empty()){
	        imgs.push_back(frame);
	    }
	 }

	 video.release();

	 Mat pano;
	 Ptr<Stitcher> stitcher = cv::Stitcher::create(mode, try_use_gpu);
	 Stitcher::Status status = stitcher->stitch(imgs, pano);

	 if (status != Stitcher::OK){
        cout << "Can't stitch images, error code = " << int(status) << endl;
        /*
        *	1 -->  Necessario  mais  imagens
        *	2 -->  Falha na estimação da matriz homografica
        *	3 -->  Falha nos parâmetros definidos da câmera.
        */
        return -1;
	 }
	 resize(pano, pano, Size(840, 480), 0, 0, INTER_LINEAR);

	 namedWindow("Resultado", WINDOW_AUTOSIZE);
	 imshow("Resultado", pano);
	 waitKey(0);

	 return 0;
}
