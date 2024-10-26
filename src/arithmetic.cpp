#include "arithmetic.h"
#include "cmath"
#include <algorithm>
#include <numeric>
using namespace std;
using namespace cyg;
/**
 * 
 * std::valarray would have been helpful since it is very suited for vectorized operations on arrays of numbers
 * but i choose to stick with vectors for simplicity & mem mngmt
 */
/**
 * lambda functions to handle basic operations for two scalars
 */
const auto op_add = [](float a, float b)
{ return a + b; };
const auto op_sub = [](float a, float b)
{ return a - b; };
const auto op_div = [](float a, float b)
{ return a / b; };
const auto op_mul = [](float a, float b)
{ return a * b; };
const auto op_pow = [](float a, float b)
{ return std::pow(a, b); };

/**
 * @brief operation to run on cpu, plan to use openmpi for some speed ups
 * operation is executed on vectors
 * @TODO - consider extending to other types such as int, uint using generics
 * 
 * @param &out(type std::vector<float>) - output parameter
 * @param &lhs(type std::vector<float>) - input parameter - vector 
 * @param &rhs(type std::vector<float>) - input parameter - vector
 * @param &op(type T) - operation to execute, should be a function that accepts two scalars(type: float) and returns a float (same type with output param)
 */
template <typename T>
void cyg::op_cpu(std::vector<float> &out, const std::vector<float> &lhs, const std::vector<float> &rhs, T op)
{
    // #pragma omp parallel for
    for (auto i = 0; i < out.size(); i++)
        out[i] = op(lhs[i], rhs[i]);
};

/**
 * operator overloading for vector, can easily delegate the underlying operations to cuda for speed up if cuda is available
 * 
 */
std::vector<float> cyg::exp(const std::vector<float>& v1)
{
    vector<float> out_data;
    out_data.reserve(v1.size());
    for(const auto&j: v1) out_data.push_back(std::exp(v1[j]));

    return out_data;
}
std::vector<float> cyg::mean(const std::vector<float>& v1, std::vector<int>& tdims, int dim)
{
    tdims.push_back(1);
    auto shift = std::accumulate(next(tdims.cbegin(), dim+1), tdims.cend(), 1, multiplies<int>());
    tdims.pop_back();
    auto nums = tdims[dim];    
    // @todo improve to o(n)
    vector<float> out_data;
    int start = 0;
    while(start<v1.size()){
        int m = start;
        for(int k=0;k<shift;k++){
            int c = 0;
            float s = 0.0;
            do{
                s+=v1[m+shift * c];
                c++;
            }
            while(c<nums);
            out_data.push_back(s/nums);
            m+=1;
        }
        start+=(shift*nums);
    }
    return out_data;
}
std::vector<float> cyg::log(const std::vector<float> &v1)
{
    vector<float> out_data;
    out_data.reserve(v1.size());
    for(const auto& j:v1) out_data.push_back(std::log(j));

    return out_data;
}
std::vector<float> cyg::pow(const std::vector<float> &v1, const std::vector<float> v2)
{
    vector<float> out_data(v1.size(),0);
    op_cpu(out_data, v1, v2, op_pow);
    return out_data;
}
std::vector<float> cyg::pow(const std::vector<float> &v1, const float v2)
{
    vector<float> v2_vec(v1.size(),0);
    return cyg::pow(v1, v2_vec);
}

std::vector<float> cyg::operator+(const std::vector<float>& v1, const std::vector<float>& v2)
{
    vector<float> out_data(v1.size(),0);
    op_cpu(out_data, v1, v2, op_add);
    return out_data;
}
std::vector<float> cyg::operator+(const std::vector<float> &v1, const float v2)
{
    vector<float> v2_vec(v1.size(),0);
    return v1 + v2_vec;
}
std::vector<float> cyg::operator-(const std::vector<float> &v1, const std::vector<float> &v2)
{
    vector<float> out_data(v1.size(),0);
    op_cpu(out_data, v1, v2, op_sub);
    return out_data;
}
std::vector<float> cyg::operator-(const std::vector<float> &v1, const float v2)
{
    vector<float> v2_vec(v1.size(),0);
    return v1 - v2_vec;
}
std::vector<float> cyg::operator*(const std::vector<float> &v1, const std::vector<float> &v2)
{
    vector<float> out_data(v1.size(),0);
    op_cpu(out_data, v1, v2, op_mul);
    return out_data;
}
std::vector<float> cyg::operator*(const std::vector<float> &v1, const float v2)
{
    vector<float> v2_vec(v1.size(),0);
    return v1 * v2_vec;
}
std::vector<float> cyg::operator/(const std::vector<float> &v1, const std::vector<float> &v2)
{
    vector<float> out_data(v1.size(),0);
    op_cpu(out_data, v1, v2, op_div);
    return out_data;
}
std::vector<float> cyg::operator/(const std::vector<float> &v1, const float v2)
{
    vector<float> v2_vec(v1.size(),0);
    return v1 / v2_vec;
}
