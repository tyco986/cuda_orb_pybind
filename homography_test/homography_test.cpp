/**
 * Homography test: compare CUDA ORB + cv::findHomography vs OpenCV ORB + cv::findHomography
 * Default images: data/img1.png, data/img2.png
 */

#include "orb.h"
#include "warmup.h"
#include "cuda_utils.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>

static void printHomography(const char* label, const cv::Mat& H)
{
	std::cout << "\n===== " << label << " =====\n";
	if (H.empty())
	{
		std::cout << "(empty - insufficient matches)\n";
		return;
	}
	std::cout << std::fixed << std::setprecision(6);
	for (int r = 0; r < H.rows; r++)
	{
		for (int c = 0; c < H.cols; c++)
			std::cout << std::setw(12) << H.at<double>(r, c) << " ";
		std::cout << "\n";
	}
}

static void runCudaOrbHomography(const cv::Mat& img1, const cv::Mat& img2)
{
	std::cout << "\n----- CUDA ORB + cv::findHomography -----\n";

	auto t0 = std::chrono::high_resolution_clock::now();

	int3 whp1, whp2;
	whp1.x = img1.cols; whp1.y = img1.rows; whp1.z = iAlignUp(whp1.x, 128);
	whp2.x = img2.cols; whp2.y = img2.rows; whp2.z = iAlignUp(whp2.x, 128);

	unsigned char* d_img1 = NULL, * d_img2 = NULL;
	size_t pitch1 = whp1.y * whp1.z * sizeof(unsigned char);
	size_t pitch2 = whp2.y * whp2.z * sizeof(unsigned char);
	CHECK(cudaMalloc(&d_img1, pitch1));
	CHECK(cudaMalloc(&d_img2, pitch2));
	CHECK(cudaMemcpy2D(d_img1, whp1.z, img1.data, img1.step, img1.cols, img1.rows, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy2D(d_img2, whp2.z, img2.data, img2.step, img2.cols, img2.rows, cudaMemcpyHostToDevice));

	orb::Orbor detector;
	detector.init(5, 31, 4, orb::HARRIS_SCORE, 31, 20, 0, 10000);

	orb::OrbData data1, data2;
	detector.initOrbData(data1, 10000, true, true);
	detector.initOrbData(data2, 10000, true, true);

	unsigned char* desc1 = NULL, * desc2 = NULL;
	detector.detectAndCompute(d_img1, data1, whp1, (void**)&desc1, true);
	detector.detectAndCompute(d_img2, data2, whp2, (void**)&desc2, true);
	detector.match(data1, data2, desc1, desc2, 0.75f);

	auto t1 = std::chrono::high_resolution_clock::now();
	double ms_orb = std::chrono::duration<double, std::milli>(t1 - t0).count();

	std::vector<cv::Point2f> pts1, pts2;
	for (int i = 0; i < data1.num_pts; i++)
	{
		int k = data1.h_data[i].match;
		if (k >= 0)
		{
			pts1.push_back(cv::Point2f((float)data1.h_data[i].x, (float)data1.h_data[i].y));
			pts2.push_back(cv::Point2f((float)data2.h_data[k].x, (float)data2.h_data[k].y));
		}
	}

	std::cout << "Matched pairs: " << pts1.size() << "\n";

	cv::Mat H;
	auto t2 = std::chrono::high_resolution_clock::now();
	if (pts1.size() >= 4)
		H = cv::findHomography(pts1, pts2, cv::RANSAC, 5.0);
	auto t3 = std::chrono::high_resolution_clock::now();
	double ms_homography = std::chrono::duration<double, std::milli>(t3 - t2).count();

	std::cout << "Time (detectAndCompute + match): " << std::fixed << std::setprecision(2) << ms_orb << " ms\n";
	std::cout << "Time (findHomography):          " << std::fixed << std::setprecision(2) << ms_homography << " ms\n";

	printHomography("CUDA ORB Homography", H);

	detector.freeOrbData(data1);
	detector.freeOrbData(data2);
	CHECK(cudaFree(desc1));
	CHECK(cudaFree(desc2));
	CHECK(cudaFree(d_img1));
	CHECK(cudaFree(d_img2));
}

static void runOpenCVOrbHomography(const cv::Mat& img1, const cv::Mat& img2)
{
	std::cout << "\n----- OpenCV ORB + cv::findHomography -----\n";

	auto t0 = std::chrono::high_resolution_clock::now();

	cv::Ptr<cv::ORB> detector = cv::ORB::create(10000);
	std::vector<cv::KeyPoint> kpt1, kpt2;
	cv::Mat desc1, desc2;
	detector->detectAndCompute(img1, cv::Mat(), kpt1, desc1);
	detector->detectAndCompute(img2, cv::Mat(), kpt2, desc2);

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
	std::vector<cv::DMatch> matches;
	matcher->match(desc1, desc2, matches);

	auto t1 = std::chrono::high_resolution_clock::now();
	double ms_orb = std::chrono::duration<double, std::milli>(t1 - t0).count();

	std::vector<cv::Point2f> pts1, pts2;
	for (const auto& m : matches)
	{
		pts1.push_back(kpt1[m.queryIdx].pt);
		pts2.push_back(kpt2[m.trainIdx].pt);
	}

	std::cout << "Matched pairs: " << pts1.size() << "\n";

	cv::Mat H;
	auto t2 = std::chrono::high_resolution_clock::now();
	if (pts1.size() >= 4)
		H = cv::findHomography(pts1, pts2, cv::RANSAC, 5.0);
	auto t3 = std::chrono::high_resolution_clock::now();
	double ms_homography = std::chrono::duration<double, std::milli>(t3 - t2).count();

	std::cout << "Time (detectAndCompute + match): " << std::fixed << std::setprecision(2) << ms_orb << " ms\n";
	std::cout << "Time (findHomography):          " << std::fixed << std::setprecision(2) << ms_homography << " ms\n";

	printHomography("OpenCV ORB Homography", H);
}

int main(int argc, char** argv)
{
	const char* path1 = (argc > 1) ? argv[1] : "data/img1.png";
	const char* path2 = (argc > 2) ? argv[2] : "data/img2.png";

	cv::Mat img1 = cv::imread(path1, cv::IMREAD_GRAYSCALE);
	cv::Mat img2 = cv::imread(path2, cv::IMREAD_GRAYSCALE);

	if (img1.empty() || img2.empty())
	{
		std::cerr << "Failed to load images: " << path1 << ", " << path2 << "\n";
		return -1;
	}

	std::cout << "Images: " << path1 << " (" << img1.cols << "x" << img1.rows << "), "
		<< path2 << " (" << img2.cols << "x" << img2.rows << ")\n";

	initDevice(0);
	warmup();

	runCudaOrbHomography(img1, img2);
	runOpenCVOrbHomography(img1, img2);

	CHECK(cudaDeviceReset());
	std::cout << "\nDone.\n";
	return 0;
}
