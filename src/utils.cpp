#include <iostream>
#include "utils.h"

using namespace std;

default_random_engine dfe(time(nullptr));

void CHECK_TRANSPOSE(std::vector<size_t> s, int a, int b)
{
    const bool isValid = (-s.size() <= a < s.size()) && (-s.size() <= b < s.size()) && (std::abs(a - b) == 1);
    assertm(isValid, ERROR_TRANSPOSE);
    if (!isValid)
        throw std::runtime_error(ERROR_TRANSPOSE);
}

void CHECK_VALID_DIMS(std::vector<size_t> dims)
{
    assertm(dims.size() != 0, ERROR_INVALID_DIMS);
    if (dims.size() == 0)
        throw std::runtime_error(ERROR_INVALID_DIMS);
    auto min_ele = *std::min_element(dims.cbegin(), dims.cend());
    assertm(min_ele > 0, ERROR_INVALID_DIMS);
    if (min_ele < 1)
        throw std::runtime_error(ERROR_INVALID_DIMS);
}

void CHECK_RANK(std::vector<size_t> dims, std::vector<size_t> tdims)
{
    assertm(dims.size() == tdims.size(), ERROR_RANK_MISMATCH);
    if (dims.size() != tdims.size())
        throw std::runtime_error(ERROR_RANK_MISMATCH);
}

void CHECK_SIZE(std::vector<size_t> dims, int n_elements)
{
    const int size_tdims = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
    assertm(n_elements == size_tdims, ERROR_SIZE_MISMATCH) if (n_elements != size_tdims) throw std::runtime_error(ERROR_SIZE_MISMATCH);
}

void CHECK_ARGS_OPS_BROADCAST(const std::vector<size_t> dims, const std::vector<size_t> tdims)
{
    for (int i = -1; i >= -std::min(dims.size(), tdims.size()); i--)
    {
        if (std::min(dims[dims.size() + i], tdims[tdims.size() + i]) != 1 && tdims[tdims.size() + i] != dims[dims.size() + i])
            throw std::runtime_error(ERROR_SIZE_MISMATCH);
    }
}

void CHECK_VALID_INDEX(std::vector<size_t> dims, std::vector<size_t> tdims)
{
    for (int i = 0; i < tdims.size(); ++i)
        if (dims[i] != INT_MAX && (dims[i] >= tdims[i] || dims[i] < 0))
            throw std::runtime_error(ERROR_OUT_OF_RANGE);
}

void CHECK_MM_DIMS(std::vector<size_t> ldims, std::vector<size_t> rdims)
{
    std::iter_swap(ldims.rbegin(), ldims.rbegin() + 1);
    int min_d = std::min(rdims.size(), ldims.size());
    bool isValid = std::equal(rdims.rbegin() + 1, rdims.rbegin() + min_d, ldims.rbegin() + 1, ldims.rbegin() + min_d);
    assertm(isValid, ERROR_MM_COMPATIBLE);
    if (!isValid)
        throw std::runtime_error(ERROR_MM_COMPATIBLE);
}

void CHECK_VALID_RANGE(const int &dim, const int &rank, const int &low)
{
    assertm(dim == INT_MAX || low <= dim < rank, ERROR_OUT_OF_BOUND_DIM);
    if (dim != INT_MAX && (dim >= rank || dim < low))
        throw std::runtime_error(ERROR_OUT_OF_BOUND_DIM);
}

void CHECK_EQUAL_SIZES(const std::vector<size_t> dims1, const std::vector<size_t> dims2)
{
    const bool isValid = dims1.size() == dims2.size() && std::equal(dims1.begin(), dims1.end(), dims2.begin());
    if (!isValid)
        throw std::runtime_error("invalid op, tensors sizes must be the same");
}

float generate_random(const float &low, const float &high)
{
    uniform_real_distribution<float> u(low, high);
    return u(dfe);
}

size_t get_index(vector<size_t> t_dims, vector<size_t> dims)
{
    // CHECK_ARGS_DIMS(dims, -1, t_dims);
    auto index = dims[dims.size() - 1];
    for (auto i = 0; i < dims.size() - 1; i++)
    {
        index += (dims[i] * t_dims[i + 1]);
    };
    return index;
}
tuple<valarray<size_t>, valarray<size_t>> generate_idxs(const vector<size_t> tdims, int dim)
{
    if (dim < 0)
        dim = tdims.size() + dim;
    valarray<size_t> strides(tdims.size());
    size_t s = 1;
    for (int i = tdims.size() - 1; i >= 0; --i)
    {
        strides[i] = s;
        s *= tdims[i];
    }
    int n_elements = strides[0] * tdims[0];
    valarray<size_t> sizes(tdims.data(), tdims.size());
    sizes[dim] = 1;
    valarray<size_t> id_data(n_elements);
    iota(begin(id_data), end(id_data), 0);
    valarray<size_t> idxs = id_data[gslice(0, sizes, strides)];

    return {strides, idxs};
}

bool is_broadcastable(const vector<size_t> &dims, const vector<size_t> &tdims)
{
    for (int i = -1; i >= -min(dims.size(), tdims.size()); i--)
    {
        if (min(dims[dims.size() + i], tdims[tdims.size() + i]) != 1 && tdims[tdims.size() + i] != dims[dims.size() + i])
            return false;
    }
    return true;
}

ostream &operator<<(ostream &out, const vector<size_t> input)
{
    out << "(";
    for (int i = 0; i < input.size() - 1; i++)
        out << input[i] << " , ";
    out << input[input.size() - 1];
    out << ")";
    return out;
};