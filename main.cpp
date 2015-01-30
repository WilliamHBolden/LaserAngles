#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"

#include <iostream>
#include <fstream>

int searchIterations = 10;

unsigned int frameWidth = 1280/2;
unsigned int frameHeight = 720/2;

//float paperWidth = 23.9; //cm
//float paperHeight = 18.1; //cm

//float numBoxW = 22;
//float numBoxH = 18;

//float dist = 20; //cm

float primaryScreenWidth;
float primaryScreenHeight;

float secondaryScreenWidth;
float secondaryScreenHeight;

struct Camera
{
	cv::VideoCapture cam;
	cv::Mat cameraMatrix;
	cv::Mat distCoeffs;

	cv::Mat transMat;

	float screenWidth;
	float screenHeight;
};


float getAngle(float distanceToScreen, float distanceFromOrigin);
void drawPoints(cv::Mat* image, const std::vector<cv::Point2f>& points);



void loadCalibration(cv::Mat& cameraMatrix, cv::Mat& distCoeffs, const char* filepath);
void loadCalibration(cv::Mat& cameraMatrix, cv::Mat& distCoeffs, int cameraSelection);

std::vector<cv::Point> getQuad( cv::Mat& image);
std::vector<cv::Point2f> findCorners(Camera& camera, int iterations);

std::vector<cv::Point2f> orientCorners(std::vector<cv::Point2f> inCorners);
std::vector<cv::Point2f> orientCorners(std::vector<cv::Point> points);
std::vector<cv::Point2f> getImageCorners(cv::Mat& image);
cv::Mat getTransformMat(cv::Mat& image, std::vector<cv::Point>& corners);
void transform(cv::Mat& image, cv::Mat& transMat);

//To be removed?
//void undistortAndCrop(cv::Mat& image, cv::Mat& cameraMatrix, cv::Mat& distCoeffs, std::vector<cv::Point>& corners);

std::vector<int> chooseCameras();

std::vector<cv::Point2f> findLasers(cv::Mat& images);

cv::Mat getImage(Camera& cam);














void loadCalibration(cv::Mat& cameraMatrix, cv::Mat& distCoeffs, const char* filepath)
{
	std::ifstream file;
	file.open(filepath);

	cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
	distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

	printf("Loading Camera matrix coefficients from %s\n", filepath);
	for(int x = 0; x < 3; x++)
	{
		for(int y = 0; y < 3; y++)
		{
			file >> cameraMatrix.at<double>(x, y);
			printf("(%d, %d): %f\n", x, y, cameraMatrix.at<double>(x, y));
		}
	}

	printf("Loading distortion coefficients from %s\n", filepath);
	for(int i = 0; i < 8; i++)
	{
		file >> distCoeffs.at<double>(i, 0);
		printf("(%d): %f\n", i, distCoeffs.at<double>(i, 0));
	}

	file.close();
}

void loadCalibration(cv::Mat& cameraMatrix, cv::Mat& distCoeffs, int cameraSelection)
{
	std::string path = "calibration/cam"+std::to_string(cameraSelection)+".cal";
	loadCalibration(cameraMatrix, distCoeffs, path.c_str());
}

float getAngle(float distanceToScreen, float distanceFromOrigin)
{
	return atanf(distanceFromOrigin/distanceToScreen); //sqrtf(distanceToScreen*distanceToScreen + distanceFromOrigin*distanceFromOrigin
}

void drawPoints(cv::Mat& image, const std::vector<cv::Point2f>& points)
{
	for(unsigned int i = 0; i < points.size(); i++)
	{
		cv::circle(image, points[i], 2, cv::Scalar(0, 0, 255), -1);
	}
}

std::vector<cv::Point> getQuad( cv::Mat& image)
{
	std::vector<cv::Point> outCorners;
	cv::Mat bw;

	cv::cvtColor(image, bw, CV_BGR2GRAY);

	//cv::blur(bw, bw, cv::Size(5,5));

	//cv::Canny(bw, bw, 166, 500, 5, true);
	cv::Canny(bw, bw, 100, 300, 5, true);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(bw, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	std::vector<cv::Point> approxShape;

	std::vector<std::vector<cv::Point>> quads;
	std::vector<double> areas;

	for(unsigned int i = 0; i < contours.size(); i++)
	{
		cv::approxPolyDP(
			contours[i], 
			approxShape, 
			cv::arcLength(contours[i], true)*0.02, 
			true);

		double area = std::fabs(cv::contourArea(contours[i]));

		if(area > image.cols*image.rows/(20) && approxShape.size() == 4)
		{
			quads.push_back(approxShape);
			areas.push_back(area);
		}
	}


	if(quads.size() != 0)
	{
		int largestQuad = 0;
		for(unsigned int i = 0; i < quads.size(); i++)
		{
			if(areas[largestQuad] < areas[i])
			{
				largestQuad = i;
			}
		}
		outCorners = quads[largestQuad];
	}
	return outCorners;
}

bool epseq(double p1, double p2, double epsilon)
{
	return (p1 > (p2 - epsilon)) && (p1 < (p2 + epsilon));
}

std::vector<cv::Point2f> findCorners(Camera& camera, int iterations)
{
	cv::Mat image;
	std::vector<std::vector<cv::Point>> corners;
	std::vector<double> areas;
	std::vector<double> modeVal;
	std::vector<std::vector<int>> indices;

	for(int i = 0; i < iterations; i++)
	{
		image = getImage(camera);

		std::vector<cv::Point> quad;
		quad = getQuad(image);

		if(quad.size() != 0)
		{
			corners.push_back(quad);
			areas.push_back(std::fabs(cv::contourArea(quad)));
		}
	}

	printf("Corners.size %d   %f\n", corners.size(), areas[0]);

	for(unsigned int i = 0;  i < areas.size(); i++)
	{
		printf("%f\n", areas[i]);
	}

	for(unsigned int x = 0; x < areas.size(); x++)
	{
		bool placed = false;
		unsigned int y = 0;
		while(y < modeVal.size() && !placed)
		{
			double epsilon = modeVal[y] * 0.05;

			if(epseq(areas[x], modeVal[y], epsilon))
			{
				indices[y].push_back(x);
				placed = true;
			}

		}

		if(!placed)
		{
			modeVal.push_back(areas[x]);
			std::vector<int> index;
			index.push_back(x);
			indices.push_back(index);
		}
	}

	int mode = 0;
	for(unsigned int i = 0; i < indices.size(); i++)
	{
		if(indices[i].size() > mode)
		{
			mode = i;
		}
	}

	cv::Point2f emptyPoint(0, 0);
	std::vector<cv::Point2f> average;

	average.push_back(emptyPoint);
	average.push_back(emptyPoint);
	average.push_back(emptyPoint);
	average.push_back(emptyPoint);

	//printf("a size   %d\n", indices[mode].size());

	for(unsigned int i = 0; i < indices[mode].size(); i++)
	{
		std::vector<cv::Point2f> point;
		point = orientCorners(corners[indices[mode][i]]);

		for( int p = 0; p < 4; p++)
		{
			average[p].x += point[p].x;
			average[p].y += point[p].y ;
		}

	}

	for(int p = 0; p < 4; p++)
	{
		average[p].x /= (float)indices[mode].size();
		average[p].y /= (float)indices[mode].size();
	}

	return average;
}


std::vector<cv::Point2f> orientCorners(std::vector<cv::Point2f> inCorners)
{

	for(int i = 1; i < 4; i++)
	{
		float tempxy = inCorners[i].x + inCorners[i].y;
		cv::Point2f temp = inCorners[i];

		int n;
		for(n = i - 1; n >= 0 && tempxy < (inCorners[n].x+inCorners[n].y); n--)
		{
			inCorners[n+1] = inCorners[n];
		}
		inCorners[n+1] = temp;
	}

	if(inCorners[1].x > inCorners[2].x)
	{
		cv::Point2f temp = inCorners[2];
		inCorners.erase(inCorners.begin()+2);
		inCorners.push_back(temp);
	}
	else
	{
		cv::Point2f temp = inCorners[1];
		inCorners.erase(inCorners.begin()+1);
		inCorners.push_back(temp);
	}

	return inCorners;
}

std::vector<cv::Point2f> orientCorners(std::vector<cv::Point> points)
{
	std::vector<cv::Point2f> inCorners;

	for(int i = 0; i < 4; i++)
	{
		inCorners.push_back(cv::Point2f(points[i].x, points[i].y));
	}
	
	return orientCorners(inCorners);
}

std::vector<cv::Point2f> getImageCorners(cv::Mat& image)
{
	std::vector<cv::Point2f> outCorners;
	outCorners.push_back(cv::Point2f(0, 0));
	outCorners.push_back(cv::Point2f(image.cols, 0));
	outCorners.push_back(cv::Point2f(image.cols, image.rows));
	outCorners.push_back(cv::Point2f(0, image.rows));

	return outCorners;
}

void transform(cv::Mat& image, cv::Mat& transMat)
{
	cv::warpPerspective(image, image, transMat, image.size());
}

cv::Mat getTransformMat(cv::Mat& image, std::vector<cv::Point2f>& corners)
{
	std::vector<cv::Point2f> inCorners = orientCorners(corners);
	std::vector<cv::Point2f> outCorners = getImageCorners(image);

	return getPerspectiveTransform(inCorners, outCorners);
}

cv::Mat getTransformMat(cv::Mat& image, std::vector<cv::Point>& corners)
{
	std::vector<cv::Point2f> inCorners = orientCorners(corners);
	std::vector<cv::Point2f> outCorners = getImageCorners(image);

	return getPerspectiveTransform(inCorners, outCorners);
}


std::vector<int> chooseCameras()
{
	std::vector<cv::VideoCapture> availableCameras;
	std::vector<int> selected;

	int numCameras = 0;
	for(;;)
	{
		cv::VideoCapture Camera(numCameras);

		if(Camera.get(CV_CAP_PROP_FRAME_WIDTH) == -1)
		{
			break;
		}

		cv::namedWindow("Camera "+std::to_string(numCameras), CV_WINDOW_AUTOSIZE);

	Camera.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	Camera.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
		availableCameras.push_back(Camera);

		numCameras++;
	}

	if(availableCameras.size() < 2)
	{
		printf("Less than 2 cameras\n");
		return selected;
	}

	int selection = -1;
	cv::Mat image;

	while(!(selected.size() == 2))
	{
		//Display
		for(int i = 0; i < numCameras; i++)
		{
			availableCameras[i] >> image;

			if(selection != i)
			{
				cv::imshow("Camera "+std::to_string(i), image);
			}
		}

		//Input
		char key = cv::waitKey(25);

		if(key >= '0')
		{
			int keyVal = atoi(&key);
			if((atoi(&key) != selection) && ((keyVal < numCameras && keyVal > -1) || !(selection < numCameras && selection > -1)))
			{
				selection = keyVal;
				selected.push_back(selection);
				cv::destroyWindow("Camera "+std::to_string(keyVal));
			}
		}

	}

	cv::destroyAllWindows();

	return selected;
}


cv::Mat getImage(Camera& cam)
{
	cv::Mat image;
	cv::Mat temp;

	cam.cam >> image;

	temp = image.clone();

	cv::undistort(temp, image, cam.cameraMatrix, cam.distCoeffs);

	return image;
}

void initTransMat(Camera& cam, int iterations)
{
	std::vector<cv::Point2f> cor = findCorners(cam, iterations);

	cv::Mat temp = getImage(cam);

	cam.transMat = getTransformMat(temp, cor);
}

struct Rect
{
	int top;
	int bottom;
	int left;
	int right;

	bool contains(int x, int y)
	{
		return (left <= x && right >= x) && (top >= y && bottom <= y);
	}
};


std::vector<cv::Point2f> findLasers(cv::Mat& image)
{
	//std::vector<std::vector<cv::Point>> contours;
	//cv::findContours(bw, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	cv::Mat bw;
	cv::cvtColor(image, bw, CV_BGR2GRAY);

	cv::inRange(bw, 220, 255, bw);

	std::vector<Rect> searchAreas;

	for(int x = 0; x < bw.rows ; x++)
	{
		for(int y = 0; y < bw.cols ; y++)
		{
			if(bw.at<unsigned char>(x, y) > 0)
			{
				bool found = false;

				for(int i = 0; i < searchAreas.size(); i++)
				{
					if(searchAreas[i].contains(x, y))
					{
						found = true;
						break;
					}
				}

				if(!found) //need to make this scale with resolution
				{
					Rect rect;
					rect.left = x - 50;
					if(rect.left < 0) rect.left = 0;

					rect.right = x + 50;
					if(rect.right > bw.rows) rect.right = bw.rows;

					rect.top = y + 100;
					if(rect.top > bw.cols) rect.top = bw.cols;

					rect.bottom = y;

					searchAreas.push_back(rect);
				}
			}
		}
	}

	std::vector<cv::Point2f> lasers;

	for(int i = 0; i < searchAreas.size(); i++)
	{
		int num = 0;
		float xs = 0;
		float ys = 0;

		for(int x = searchAreas[i].left; x < searchAreas[i].right ; x++)
		{
			for(int y = searchAreas[i].bottom; y < searchAreas[i].top ; y++)
			{
				if(bw.at<unsigned char>(x, y) > 0)
				{
					num++;
					xs += x;
					ys += y;
				}
			}
		}

		xs /= num;
		ys /= num;

		cv::Point2f laser(ys, xs);

		lasers.push_back(laser);
	}

	return lasers;
}

cv::Point2f getPositionCM(cv::Point2f& point, Camera& cam, cv::Mat& image)
{
	float cmPerPixelX = cam.screenWidth/image.cols;
	float cmPerPixelY = cam.screenHeight/image.rows;

	float xPixelPos = point.x - image.cols/2.0f;
	float yPixelPos = -(point.y - image.rows/2.0f);

	float cmX = xPixelPos * cmPerPixelX;
	float cmY = yPixelPos * cmPerPixelY;

	return cv::Point2f(cmX, cmY);
}

int main()
{

	//Get Cameras (user) or error
	std::vector<int> cameraIds = chooseCameras();

	cv::namedWindow ("Primary", CV_WINDOW_AUTOSIZE);
	cv::namedWindow ("Secondary", CV_WINDOW_AUTOSIZE);

	Camera primary;
	Camera secondary;


	printf("%d, %d\n", cameraIds[0], cameraIds[1] );

	std::ifstream file;
	file.open("config");

	file >> frameWidth;
	file >> frameHeight;

	file >> primary.screenWidth;
	file >> primary.screenHeight;

	file >> secondary.screenWidth;
	file >> secondary.screenHeight;

	file >> searchIterations;

	file.close();

	printf("Primary screen dimensons: %fcm, %fcm\n", primary.screenWidth, primary.screenHeight);
	printf("Secondary screen dimensons: %fcm, %fcm\n", secondary.screenWidth, secondary.screenHeight);

	primary.cam = cv::VideoCapture(cameraIds[0]);
	primary.cam.set(CV_CAP_PROP_FRAME_WIDTH, frameWidth);
	primary.cam.set(CV_CAP_PROP_FRAME_HEIGHT, frameHeight);

	secondary.cam = cv::VideoCapture(cameraIds[1]);
	secondary.cam.set(CV_CAP_PROP_FRAME_WIDTH, frameWidth);
	secondary.cam.set(CV_CAP_PROP_FRAME_HEIGHT, frameHeight);


	loadCalibration(primary.cameraMatrix, primary.distCoeffs, cameraIds[0]);
	loadCalibration(secondary.cameraMatrix, secondary.distCoeffs, cameraIds[1]);



	initTransMat(primary, searchIterations);
	//initTransMat(secondary, searchIterations);


	char key = 0;

	cv::Mat image;

	while(key != 'q')
	{
		image = getImage(primary);
		transform(image,  primary.transMat);

		//cv::Mat bw;
		//cv::cvtColor(image, bw, CV_BGR2GRAY);

		//cv::inRange(bw, 220, 255, bw);

		std::vector<cv::Point2f> plasers = findLasers(image);
		drawPoints(image, plasers);

		cv::imshow("Primary", image);



		
		image = getImage(secondary);
		transform(image,  secondary.transMat);



		cv::imshow("Secondary", image);
		
		std::vector<cv::Point2f> slasers = findLasers(image);
		drawPoints(image, slasers);

		key = cv::waitKey(25);
	}



	//Choose primary Camera
	//load distortion correction
	//find view rectangle

	//track laser
		//each frame apply undistort and perspective transform

}