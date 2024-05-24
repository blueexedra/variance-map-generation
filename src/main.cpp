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
#include <filesystem>

#include "CppProgressBar.hpp"
#include "VarianceComputer.h"

using namespace std;
using namespace cv;
using namespace util;

int main(int argc, char** argv)
{
	if (argc != 3)
	{
		cout << "Usage: variance_map [image_path] [window_size]" << endl;
		return 1;
	}

	filesystem::path data_path(argv[1]);

	vector<Mat_<float>> images;
	filesystem::path extension;
	vector<filesystem::path> image_names;

	cout << "Loading images..." << endl;
	for(const auto& f : filesystem::directory_iterator(data_path)){
		const auto path = f.path();
		const auto image = imread(path, IMREAD_GRAYSCALE);
		if (image.empty())
		{
			continue;
		}

		extension = path.extension();
		images.emplace_back(image);
		image_names.emplace_back(path.stem());
	}

	const auto number_of_images = images.size();
	cout << "Loaded " << number_of_images << " images" << endl;
	cout << endl;

	vector<size_t> widths(images.size());
	vector<size_t> heights(images.size());

	for (size_t i = 0; i < number_of_images; i++)
	{
		widths[i] = images[i].cols;
		heights[i] = images[i].rows;
	}

	const size_t window_size = strtol(argv[2], nullptr, 10);
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
				VarianceComputer computer(window_size, images[count]);
				varianceMap.at<float>(v, u) = computer.computeVarianceAt(make_pair(u, v));
			}
		}
		varianceMaps.emplace_back(varianceMap);
		count++;
	};

	cout << "[Generate Variance Map]" << endl;
	for_progress(number_of_images, lambdaBody);

	count = 0;
	data_path /= "variances";
	filesystem::create_directories(data_path);

	auto lambdaBodyForWriting = [&](string& output_string)
	{
		const auto name = image_names[count];
		std::string output_path(data_path.string());
		output_path += "/";
		output_path += name;
		output_path += "_variance";
		output_path += "_window";
		output_path += to_string(window_size);
		output_path += extension.string();

		imwrite(output_path, varianceMaps[count]);
		count++;
	};
	cout << "Writing images..." << endl;
	for_progress(number_of_images, lambdaBodyForWriting);
	return 0;
}