/*
The for_progress() function is:

Copyright (c) 2019 ysuzuki19

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <omp.h>

#include "CppProgressBar.hpp"
#include "VarianceComputer.h"

using namespace std;
using namespace cv;
using namespace util;

int main(int argc, char** argv)
{
	if (argc != 5)
	{
		cout << "Usage: variance_map [image_path] [number_of_images] [image_extension] [window_size]" << endl;
		return 1;
	}

	char imagePath[256];
	const size_t numberOfImages = strtol(argv[2], nullptr, 10);
	const string imageExtension(argv[3]);

	string dataPath(argv[1]);
	const string imagePrefix("im_");
	const char directorySeparatorChar = '/';
	dataPath += directorySeparatorChar;
	dataPath += imagePrefix;

	vector<Mat_<float>> images;
	cout << "Loading images..." << endl;
	for (size_t i = 0; i < numberOfImages; i++)
	{
		sprintf(imagePath, "%s%04lu.%s", dataPath.c_str(), i + 1, imageExtension.c_str());
		images.emplace_back(imread(imagePath, IMREAD_GRAYSCALE));
		if (images[i].empty())
		{
			cout << "Could not read image No." << to_string(i + 1) << endl;
			return 1;
		}
	}

	cout << "Loaded " << images.size() << " images" << endl;
	cout << endl;

	vector<size_t> widths(images.size());
	vector<size_t> heights(images.size());

	for (size_t i = 0; i < numberOfImages; i++)
	{
		widths[i] = images[i].cols;
		heights[i] = images[i].rows;
	}

	const size_t windowSize = strtol(argv[4], nullptr, 10);
	vector<Mat_<float>> varianceMaps;

	size_t count = 0;
	auto lambdaBody = [&](string& output_string)
	{
		Mat_<float> varianceMap = Mat_<float>::zeros(heights[count], widths[count]);
#pragma omp parallel for
		for (size_t u = 0; u < widths[count]; u++)
		{
			for (size_t v = 0; v < heights[count]; v++)
			{
				VarianceComputer computer(windowSize, images[count]);
				varianceMap.at<float>(v, u) = computer.computeVarianceAt(make_pair(u, v));
			}
		}
		varianceMaps.emplace_back(varianceMap);
		count++;
	};

	cout << "[Generate Variance Map]" << endl;
	for_progress(numberOfImages, lambdaBody);

	count = 0;
	auto lambdaBodyForWriting = [&](string& output_string)
	{
		sprintf(imagePath, "%s%04lu.variance.%s", dataPath.c_str(), count + 1, imageExtension.c_str());
		imwrite(imagePath, varianceMaps[count]);
		count++;
	};
	cout << "Writing images..." << endl;
	for_progress(numberOfImages, lambdaBodyForWriting);
	return 0;
}