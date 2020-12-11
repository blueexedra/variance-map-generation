#include "VarianceComputer.h"
#include "StatisticUtility.h"

namespace util
{
	VarianceComputer::VarianceComputer(std::size_t windowSize, const cv::Mat_<float>& image) : _windowSize(windowSize),
																							   _image(image)
	{
	}

	VarianceComputer::~VarianceComputer() = default;

	float VarianceComputer::computeVarianceAt(const std::pair<std::size_t, std::size_t>& pointOnImage)
	{
		const auto u = pointOnImage.first, v = pointOnImage.second;
		const auto width = _image.cols, height = _image.rows;
		const auto halfSizeOfWindow = static_cast<int>(floor(1.0 * _windowSize / 2));

		auto ave = 0.0f, var = 0.0f;
		auto count = 0;
		for (int v2 = -halfSizeOfWindow; v2 <= halfSizeOfWindow; v2++)
		{
			if (v + v2 < 0 || height <= v + v2) continue;
			for (int u2 = -halfSizeOfWindow; u2 <= halfSizeOfWindow; u2++)
			{
				if (u + u2 < 0 || width <= u + u2) continue;
				const auto v3 = v + v2, u3 = u + u2;
				const auto refPixelVal = ((float*)_image.data)[v3 * width + u3];
				ave += refPixelVal;
				var += refPixelVal * refPixelVal;
				count++;
			}
		}
		ave /= count;
		var = var / count - ave * ave;
		return var;
	}
} // namespace util