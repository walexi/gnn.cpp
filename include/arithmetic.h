#ifndef ARITHMETIC_H
#define ARITHMETIC_H
#include <vector>

namespace cyg
{
    // template<typename T>
    /**
     * @todo 
     * use template specialization to reduce code here especially for arithmetic operator overloading
     */
    template <typename T>
     void op_cpu(std::vector<float>& out, const std::vector<float>& lhs, const std::vector<float>& rhs, T t);

    std::vector<float> pow(const std::vector<float>& v1, const float v2);
    std::vector<float> pow(const std::vector<float>& v1, const std::vector<float> v2);
    std::vector<float> operator+(const std::vector<float>& v1, const std::vector<float>& v2);
    std::vector<float> operator+(const std::vector<float>& v1, const float v2);
    std::vector<float> operator-(const std::vector<float>& v1, const std::vector<float>& v2);
    std::vector<float> operator-(const std::vector<float>& v1, const float v2);
    std::vector<float> operator*(const std::vector<float>& v1, const std::vector<float>& v2);
    std::vector<float> operator*(const std::vector<float>& v1, const float v2);
    std::vector<float> operator/(const std::vector<float>& v1, const std::vector<float>& v2);
    std::vector<float> operator/(const std::vector<float>& v1, const float v2);
}
#endif