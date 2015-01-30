#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"

#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <fstream>
#include <string>

#include <time.h>

const char* WINDOW = "Calibration";

const int SCREEN_WIDTH = 1280;
const int SCREEN_HEIGHT = 720;

const int BOARD_WIDTH = 9;
const int BOARD_HEIGHT = 6;
const cv::Size boardSize(BOARD_WIDTH, BOARD_HEIGHT);

const float delay = 1.0f; //seconds

void drawPoints(cv::Mat* image, const std::vector<cv::Point2f>& points)
{
	for(unsigned int i = 0; i < points.size(); i++)
	{
		cv::circle(*image, points[i], 2, cv::Scalar(0, 0, 255), -1);
	}
}

int chooseCamera()
{
	std::vector<cv::VideoCapture> availableCameras;

	int numCameras = 0;
	for(;;)
	{
		cv::VideoCapture camera(numCameras);

		if(camera.get(CV_CAP_PROP_FRAME_WIDTH) == -1)
		{
			break;
		}

		cv::namedWindow("Camera "+std::to_string(numCameras), CV_WINDOW_AUTOSIZE);
		availableCameras.push_back(camera);

		numCameras++;
	}

	int selection = -1;
	cv::Mat image;

	while(!(selection < numCameras && selection > -1))
	{
		for(int i = 0; i < numCameras; i++)
		{
			availableCameras[i] >> image;
			cv::imshow("Camera "+std::to_string(i), image);
		}

		char key = cv::waitKey(25);

		if(key >= '0')
		selection = atoi(&key);
	}

	cv::destroyAllWindows();

	return selection;
}
 
void runCaliration(cv::Mat& cameraMatrix, cv::Mat& distCoeffs, int cameraSelection)
{
 	cv::VideoCapture capture(cameraSelection);

	capture.set(CV_CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT);

 	cv::namedWindow("Calibration", CV_WINDOW_AUTOSIZE);

 	cv::Mat image;
	char key = 0;

	std::vector<std::vector<cv::Point2f>> patterns;

	clock_t lastTime = 0;

	while(key != 'q')
	{
		capture >> image;
		std::vector<cv::Point2f> corners;

		bool success = cv::findChessboardCorners(image, boardSize, corners, 
													CV_CALIB_CB_ADAPTIVE_THRESH | 
													CV_CALIB_CB_FAST_CHECK | 
													CV_CALIB_CB_NORMALIZE_IMAGE);

		if(success && (clock() - lastTime > delay*CLOCKS_PER_SEC))
		{
			printf("Pattern Recorded\n");

			lastTime = clock();

			cv::Mat gray;
			cv::cvtColor(image, gray, CV_BGR2GRAY);
			cv::cornerSubPix( gray, corners, cv::Size(5,5), cv::Size(-1,-1), cv::TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));

			cv::drawChessboardCorners( image, boardSize, corners, success);

			patterns.push_back(corners);
		}

		cv::imshow("Calibration", image);

		key = cv::waitKey(25);
	}

	cv::destroyAllWindows();

	for(unsigned int i = 0; i < patterns.size(); i++)
	{
		printf("Set %d\n", i);
		for(unsigned int j = 0; j < patterns[i].size(); j++)
		{
			printf("%f, %f\n", patterns[i][j].x, patterns[i][j].y);
		}

		printf("\n");
	}

	std::vector<std::vector<cv::Point3f>> objects;
	std::vector<cv::Point3f> boardPattern;

	for(int y = 0; y < boardSize.height; y++)
	{
		for(int x = 0; x < boardSize.width; x++)
		{
			boardPattern.push_back(cv::Point3f(x, y, 0));
		}
	}

	objects.push_back(boardPattern);
	objects.resize(patterns.size(), objects[0]); //fill it

	cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
	distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

	std::vector<cv::Mat> rvec;
	std::vector<cv::Mat> tvec;

	double err = cv::calibrateCamera(objects, patterns, cv::Size(SCREEN_WIDTH, SCREEN_HEIGHT), cameraMatrix, distCoeffs, rvec, tvec);

	printf("Reprojction Error: %f\n", err);
}

void compareResult(cv::Mat& cameraMatrix, cv::Mat& distCoeffs, int cameraSelection)
{
	cv::VideoCapture capture(cameraSelection);

	capture.set(CV_CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT);

	cv::namedWindow("Original", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("Processed", CV_WINDOW_AUTOSIZE);

	char key = 0;

	cv::Mat image;
	while(key != 'q')
	{
		capture >> image;

		cv::imshow("Original", image);

		cv::Mat temp = image.clone();

		cv::undistort(temp, image, cameraMatrix, distCoeffs);

		cv::imshow("Processed", image);

		key = cv::waitKey(25);
	}
}

void saveCalibration(cv::Mat& cameraMatrix, cv::Mat& distCoeffs, int cameraSelection)
{
	std::ofstream file;
	file.open("cam"+std::to_string(cameraSelection)+".cal");

	for(int x = 0; x < 3; x++)
	{
		for(int y = 0; y < 3; y++)
		{
			file << cameraMatrix.at<double>(x, y) << std::endl;
		}
	}

	for(int i = 0; i < 8; i++)
	{
		file << distCoeffs.at<double>(i, 0) << std::endl;
	}

	file.close();
}

void loadCalibration(cv::Mat& cameraMatrix, cv::Mat& distCoeffs, const char* filepath)
{
	std::ifstream file;
	file.open(filepath);

	cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
	distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

	for(int x = 0; x < 3; x++)
	{
		for(int y = 0; y < 3; y++)
		{
			file >> cameraMatrix.at<double>(x, y);
		}
	}

	for(int i = 0; i < 8; i++)
	{
		file >> distCoeffs.at<double>(i, 0);
	}

	file.close();
}

int main()
{
	int selection = chooseCamera();
	cv::Mat camMat;
	cv::Mat distCoeffs;

	runCaliration(camMat, distCoeffs, selection);

	compareResult(camMat, distCoeffs, selection);

	saveCalibration(camMat, distCoeffs, selection);



//	loadCalibration(camMat, distCoeffs, "cam0.cal");

	//saveCalibration(camMat, distCoeffs, 1);

	return 0;
}