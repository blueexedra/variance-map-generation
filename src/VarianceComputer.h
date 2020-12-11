#ifndef VARIANCE_COMPUTER_H
#define VARIANCE_COMPUTER_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <utility>

namespace util
{
	class VarianceComputer
	{
	private:
		const std::size_t _windowSize;
		const cv::Mat_<float>& _image;

	public:
		VarianceComputer(std::size_t windowSize, const cv::Mat_<float>& image);

		~VarianceComputer();

		float computeVarianceAt(const std::pair<std::size_t, std::size_t>& pointOnImage);
	};
} // namespace util

#endif //VARIANCE_COMPUTER_H