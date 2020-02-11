#ifndef STATISTIC_UTILITY_H
#define STATISTIC_UTILITY_H

#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>

namespace util
{
class StatisticUtility
{
private:
    StatisticUtility();

public:
    template <typename T>
    static T max(std::vector<T> &data);

    template <typename T>
    static T min(std::vector<T> &data);

    template <typename T>
    static T sum(std::vector<T> &data, T defaultValue);

    template <typename T>
    static T mean(std::vector<T> &data, T defaultValue);

    template <typename T>
    static T variance(std::vector<T> &data, T defaultValue);
};

// Implementation

template <typename T>
T StatisticUtility::max(std::vector<T> &data)
{
    return *std::max_element(std::begin(data), std::end(data));
}

template <typename T>
T StatisticUtility::min(std::vector<T> &data)
{
    return *std::min_element(std::begin(data), std::end(data));
}

template <typename T>
T StatisticUtility::sum(std::vector<T> &data, T defaultValue)
{
    return std::accumulate(std::begin(data), std::end(data), defaultValue);
}

template <typename T>
T StatisticUtility::mean(std::vector<T> &data, T defaultValue)
{
    return sum(data, defaultValue) / data.size();
}

template <typename T>
T StatisticUtility::variance(std::vector<T, std::allocator<T>> &data, T defaultValue)
{
    T ave = defaultValue, var = defaultValue;
    for (const T &x : data)
    {
        ave += x;
        var += x * x;
    }
    ave /= data.size();
    var = var / data.size() - ave * ave;
    return var;
}
} // namespace util

#endif //STATISTIC_UTILITY_H