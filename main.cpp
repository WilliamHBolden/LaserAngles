#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"

#include <iostream>
#include <fstream>

unsigned int frameWidth = 1280;
unsigned int frameHeight = 720;

float paperWidth = 23.9; //cm
float paperHeight = 18.1; //cm

float numBoxW = 22;
float numBoxH = 18;

float dist = 20; //cm


float getAngle(float distanceToScreen, float distanceFromOrigin)
{
	return atanf(distanceFromOrigin/distanceToScreen); //sqrtf(distanceToScreen*distanceToScreen + distanceFromOrigin*distanceFromOrigin
}

void drawPoints(cv::Mat* image, const std::vector<cv::Point2f>& points)
{
	for(unsigned int i = 0; i < points.size(); i++)
	{
		cv::circle(*image, points[i], 2, cv::Scalar(0, 0, 255), -1);
	}
}

void transform(cv::Mat* image, const std::vector<cv::Point>& inPoints)
{
	std::vector<cv::Point2f> inCorners;

	for(int i = 0; i < 4; i++)
	{
		inCorners.push_back(cv::Point2f(inPoints[i].x, inPoints[i].y));
	}

	/*
		This is just sorting the points so the transform maintains the correct orientation
	*/

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

	std::vector<cv::Point2f> outCorners;
	outCorners.push_back(cv::Point2f(0, 0));
	outCorners.push_back(cv::Point2f((*image).cols, 0));
	outCorners.push_back(cv::Point2f((*image).cols, (*image).rows));
	outCorners.push_back(cv::Point2f(0, (*image).rows));

	cv::Mat trans = cv::getPerspectiveTransform(inCorners, outCorners);

	cv::warpPerspective(*image, *image, trans, (*image).size());
}

std::vector<cv::Point> getCorners( cv::Mat& image)
{
	std::vector<cv::Point> outCorners;
	cv::Mat bw;

	cv::cvtColor(image, bw, CV_BGR2GRAY);

	cv::blur(bw, bw, cv::Size(5,5));

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

/*
	locates the paper and does a perspective transform
*/
bool cropPaper(cv::Mat* image)
{
	std::vector<cv::Point> outCorners;

	outCorners = getCorners(*image);

	if(outCorners.size() == 4)
	{
		transform(image, outCorners);
		return true;
	}
	else
	{
		return false;
	}
}

cv::Point2f findLaser(cv::Mat& image)
{
	cv::Point2f loc(-1, -1);
	cv::Mat bw;
	cv::cvtColor(image, bw, CV_BGR2GRAY);

	cv::inRange(bw, 220, 255, bw);

	cv::imshow("Original", bw);

	int num = 0;
	int xs = 0;
	int ys = 0;

	for(int x = 0; x < bw.rows ; x++)
	{
		for(int y = 0; y < bw.cols ; y++)
		{
			if(bw.at<unsigned char>(x, y) > 0)
			{
				num++;
				xs += x;
				ys += y;
			}
		}
	}
	if(num > 0)
	{
		xs /= num;
		ys /= num;

		loc.x = ys;
		loc.y = xs;
	}


	return loc;
}


int main()
{
	cv::namedWindow("Original", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("Processed", CV_WINDOW_AUTOSIZE);

	cv::VideoCapture capture(0);

	if(!capture.isOpened())
	{
		std::cout << "No camera found" << std::endl;
		return 1;
	}

	capture.set(CV_CAP_PROP_FRAME_WIDTH, frameWidth);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, frameHeight);


	cv::Mat defaultImage(capture.get(CV_CAP_PROP_FRAME_HEIGHT), capture.get(CV_CAP_PROP_FRAME_WIDTH), CV_8UC3, cv::Scalar(0));
	cv::putText(defaultImage, "No surface found", cv::Point(50, defaultImage.rows/2), 4, 1.6, cv::Scalar(255, 255, 255), 3, 8); 

	std::cout << defaultImage.cols << std::endl;
	std::cout << defaultImage.rows << std::endl;


	char key = 0;
	cv::Mat image;

	std::ofstream filex;
	std::ofstream filey;
	filex.open("x2.txt");
	filey.open("y2.txt");

	while(key != 'q')
	{
		double t = cv::getTickCount();

		capture >> image;
		cv::imshow("Original", image);


		if(cropPaper(&image))
		{
			cv::Point2f laserCoord;
			laserCoord = findLaser(image);

			cv::circle(image, laserCoord, 5, cv::Scalar(0, 0, 255), -1);

			float cmPerPixelX = paperWidth/image.cols;
			float cmPerPixelY = paperHeight/image.rows;

			float xPixelPos = laserCoord.x - image.cols/2;
			float yPixelPos = -(laserCoord.y - image.rows/2);

			float cmX = xPixelPos * cmPerPixelX;
			float cmY = yPixelPos * cmPerPixelY;

			
		//	std::string xs("");
		//	xs+=cmX;

		//	std::string ys("");
		//	ys+=cmY;
			

			char xstr[200];
			char ystr[200];

			sprintf(xstr, "X: %fcm", cmX);
			sprintf(ystr, "Y: %fcm", cmY);

		//	float d = sqrtf(dist*dist + cmX*cmX);
		//	float xTheta = getAngle(d, cmX);
		//	float yTheta = getAngle(d, cmY);

			if(laserCoord.x < 0 )
			{
				cv::putText(image, "Laser not found", cv::Point(10, 30), 4, 1.0, cv::Scalar(255, 255, 255), 3, 8); 
			}
			else
			{
				cv::putText(image, xstr, cv::Point(10, 30), 4, 1.0, cv::Scalar(255, 255, 255), 3, 8); 
				cv::putText(image, ystr, cv::Point(10, 60), 4, 1.0, cv::Scalar(255, 255, 255), 3, 8); 

				filex << cmX << "\n";
				filey << cmY << "\n";
			}


			cv::imshow("Processed", image);
		}
		else
		{
			cv::imshow("Processed", defaultImage);
		}


		t = (cv::getTickCount() - t)/cv::getTickFrequency();

	//	std::cout << 1.0/t << std::endl;
		key = cv::waitKey(25);
	}
	filey.close();
	filex.close();

	cv::waitKey(0);
	return 0;
}