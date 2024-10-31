#ifndef UTIL_H
#define UTIL_H
#include <algorithm>
#include <numeric>

#include <assert.h>

#define NDEBUG
#include <cassert>

// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp));
// error messages
const char ERROR_GRAD_DTYPE[] = "Only Tensors of floating point dtype can require gradients";
const char WARNING_GRAD_NOT_LEAF[] = "UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won\'t be populated during autograd.backward()";
const char ERROR_IN_PLACE_OP_LEAF[] = "RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.";
const char ERROR_SIZE_MISMATCH[] = "tensors must be of same shape/size";
const char ERROR_RANK_MISMATCH[] = "tensors are of different ranks";
const char ERROR_OUT_OF_RANGE[] = "out of bound range";
const char ERROR_INVALID_DIMS[] = "dims cannot be empty or zero";
const char ERROR_NON_SCALAR_BACKPROP[] = "pass in tensor to backprop on non-scalar tensor";
const char ERROR_MM_COMPATIBLE[] = "tensors are not compatible, tensors should of shape [..., A] and [A ,...]";
const char ERROR_OUT_OF_BOUND_DIM[] = "dim is out of range";
const char ERROR_GRAD_MISMATCH[] = "size mismatch, incoming gradient must be same dimension with tensor";
/**
 *
 * @brief to check inputs to an inplace binary operator
 *
 * @param lhs(type std::shared_ptr<T>)
 * @param rhs(type std::shared_ptr<T>)
 */
template <class T>
void CHECK_ARGS_IN_PLACE_OPS(const std::shared_ptr<T> &lhs, const std::shared_ptr<T> &rhs)
{
    assertm(lhs->requires_grad == false, ERROR_IN_PLACE_OP_LEAF);
    if (lhs->require_grad())
        throw std::runtime_error(ERROR_IN_PLACE_OP_LEAF);
};

/**
 * @brief check valid dims
 *
 * @param dims(type std::vector<int>)
 */
inline void CHECK_VALID_DIMS(std::vector<size_t> dims)
{
    assertm(dims.size() != 0, ERROR_INVALID_DIMS);
    if (dims.size() == 0)
        throw std::runtime_error(ERROR_INVALID_DIMS);
    auto min_ele = *std::min_element(dims.cbegin(), dims.cend());
    assertm(min_ele > 0, ERROR_INVALID_DIMS);
    if (min_ele < 1)
        throw std::runtime_error(ERROR_INVALID_DIMS);
}

/**
 * @brief check rank of given dims
 *
 * @param dims(type std::vector<int>)
 * @param tdims(type std::vector<int>)
 */
inline void CHECK_RANK(std::vector<size_t> dims, std::vector<size_t> tdims)
{
    assertm(dims.size() == tdims.size(), ERROR_RANK_MISMATCH);
    if (dims.size() != tdims.size())
        throw std::runtime_error(ERROR_RANK_MISMATCH);
}

/**
 * @brief check tensor's data size given input dims
 *
 * @param dims(type std::vector<int>)
 * @param n_elements(type int)
 */
inline void CHECK_SIZE(std::vector<size_t> dims, int n_elements)
{
    const int size_tdims = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
    assertm(n_elements == size_tdims, ERROR_SIZE_MISMATCH) if (n_elements != size_tdims) throw std::runtime_error(ERROR_SIZE_MISMATCH);
}

/**
 * @brief check valid dims for indexing a tensor given the tensor's dims
 *
 * @param dims(type std::vector<int>)
 * @param tdims(type std::vector<int>)
 */
inline void CHECK_VALID_INDEX(std::vector<size_t> dims, std::vector<size_t> tdims)
{
    for (int i = 0; i < tdims.size(); ++i)
        if (dims[i] >= tdims[i] || dims[i] < 0)
            throw std::runtime_error(ERROR_OUT_OF_RANGE);
}

/**
 * @brief check valid dims for indexing a tensor given the tensor's dims
 *
 * @param dims(type std::vector<int>)
 * @param tdims(type std::vector<int>)
 */
inline void CHECK_MM_DIMS(std::vector<size_t> dims, std::vector<size_t> tdims)
{
    assertm(dims[-1] == tdims[0], ERROR_MM_COMPATIBLE);
    if (dims[-1] != tdims[0])
        throw std::runtime_error(ERROR_MM_COMPATIBLE);
}

/**
 * @brief check valid dim given the rank of a tensor
 *
 * @param dims(type std::vector<int>)
 * @param tdims(type std::vector<int>)
 */
inline void CHECK_VALID_RANGE(const int& dim, const int& rank, const int& low=0)
{
    assertm(low <= dim < rank, ERROR_OUT_OF_BOUND_DIM);
    if (dim >= rank || dim < low) throw std::runtime_error(ERROR_OUT_OF_BOUND_DIM);
}

/**
 * @author olawale onabola
 * @brief generate ids 
 */

inline std::tuple<std::valarray<std::size_t>, std::valarray<std::size_t>, std::valarray<std::size_t>> generate_idxs(const std::vector<std::size_t> tdims, const int &n_elements, const int &dim)
{
    std::valarray<std::size_t> strides(tdims.size());
    std::size_t s = 1;
    for (int i = tdims.size() - 1; i >= 0; --i)
    {
        strides[i] = s;
        s *= tdims[i];
    }
    std::valarray<std::size_t> sizes(tdims.data(), tdims.size());
    sizes[dim] = 1;
    std::valarray<std::size_t> id_data(n_elements);
    std::iota(std::begin(id_data), std::end(id_data), 0);
    const std::valarray<std::size_t> idxs = id_data[std::gslice(0, sizes, strides)];

    return {strides, sizes, idxs};
}
#endif