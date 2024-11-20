#ifndef UTIL_H
#define UTIL_H
#include <algorithm>
#include <numeric>
#include <valarray>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <string>
#include <assert.h>
#include <sstream>
#include <random>
#include <iomanip>
#define NDEBUG
#include <cassert>

// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp));

// error messages
const char ERROR_GRAD_DTYPE[] = "Only Tensors of floating point dtype can require gradients";
const char WARNING_GRAD_NOT_LEAF[] = "UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won\'t be populated during autograd.backward()";
const char ERROR_IN_PLACE_OP_LEAF[] = "RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.";
const char ERROR_SIZE_MISMATCH[] = "tensors must be of same shape/size - mismatch between number of elements and dimension of tensor";
const char ERROR_RANK_MISMATCH[] = "tensors are of different ranks";
const char ERROR_OUT_OF_RANGE[] = "out of bound range";
const char ERROR_INVALID_DIMS[] = "dims cannot be empty or zero";
const char ERROR_NON_SCALAR_BACKPROP[] = "pass in tensor to backprop on non-scalar tensor";
const char ERROR_MM_COMPATIBLE[] = "tensors are not compatible, tensors should of shape [...,A,B] and [...,B,A]";
const char ERROR_OUT_OF_BOUND_DIM[] = "dim is out of range";
const char ERROR_GRAD_MISMATCH[] = "size mismatch, incoming gradient must be same dimension with tensor";
const char ERROR_TRANSPOSE[] = "invalid inp";

std::default_random_engine e(std::time(nullptr));

std::ostream &operator<<(std::ostream &out, const std::vector<size_t> input)
{
    out << "(";
    for (int i = 0; i < input.size() - 1; i++)
        out << input[i] << " , ";
    out << input[input.size() - 1];
    out << ")";
    return out;
};

/**
 * @brief generate a floating-point number from a uniform dist in the range [low, high)
 * pdf = 1/(high-low)
 *
 * @param low(type int)
 * @param high(type int)
 *
 * @return random number(type float)
 */
float generate_random(const float &low, const float &high)
{
    std::uniform_real_distribution<float> u(low, high);
    return u(e);
}
/**
 * @brief use to create am array with the input dims and value,
 *
 * @param dims(type std::array<int, N>) //@todo array
 * @param value(type int)
 *
 * @return unique_ptr of to an array of type T(type std::unique_ptr<std::valarray<T>>)
 */
template <class T>
std::valarray<T> *initialize(std::vector<size_t> dims, T value = 1)
{
    auto n_elements = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
    auto data = new std::valarray<T>(n_elements);
    std::fill_n(std::begin(*data), n_elements, value);

    return data;
};

/**
 * @brief calculate the index of an element in a flatten array given the ND dims/cordinates
 *
 * @param t_dims(type std::valarray<int>) N_D dims of the tensor
 * @param dims(type std::valarray<int>) N_D dims of the element
 *
 * @return the 1D index of the element (type int)
 * */
auto get_index(std::vector<std::size_t> *t_dims, std::vector<std::size_t> dims)
{
    // CHECK_ARGS_DIMS(dims, -1, t_dims);
    std::transform(dims.cbegin(), dims.cend(), next(t_dims->cbegin()), dims.begin(), std::multiplies<std::size_t>());
    auto index = std::accumulate(dims.cbegin(), dims.cend(), dims[dims.size() - 1]);
    return index;
};

/**
 * @brief generate start ids along a dimension
 * for ex
 * given a 5x4 matrix
 *           [ 1,  2,  3, 4  ]
 *           [ 5,  6,  7, 8  ]
 *           [ 9, 10, 11, 12 ]
 *           [ 13,14, 15, 16 ]
 *           [ 17,18, 19, 20 ]
 *
 * start ids along the row with size of 4, are = 1, 5, 9, 13, 17
 *
 * start ids along the col with size of 5 are = 1, 2, 3, 4
 *
 * and this can be extended to more than 2D, N1xN2xN3x..xN,  N dimensions
 *
 * for 2x5x4
 *           [[[ 1,   2,  3, 4  ]
 *              [ 5,   6,  7, 8  ]
 *             [ 9,  10, 11, 12 ]
 *             [ 13, 14, 15, 16 ]
 *             [ 17, 18, 19, 20 ]]
 *
 *            [[ 21, 22, 23, 24 ]
 *             [ 25, 26, 27, 28 ]
 *             [ 29, 30, 31, 32 ]
 *             [ 33, 34, 35, 36 ]
 *             [ 37, 38, 39, 40 ]]]
 *
 * start ids across the col with size of 4, has 2*5 elements and are = 1, 5, 9, 13, 17, 21, 25, 29, 33, 37
 * start ids across the row with size of 5, has 2*4 elements and are = 1, 2, 3, 4, 21, 22, 23, 24
 * start ids across the batch with size of 2 has 5*4 elements and are =  1,2,3,4,5,6,7,8,9,....,19,20
 */

inline std::tuple<std::valarray<std::size_t>, std::valarray<std::size_t>> generate_idxs(const std::vector<std::size_t> tdims, int dim)
{
    if (dim < 0)
        dim = tdims.size() + dim;
    std::valarray<std::size_t> strides(tdims.size());
    std::size_t s = 1;
    for (int i = tdims.size() - 1; i >= 0; --i)
    {
        strides[i] = s;
        s *= tdims[i];
    }
    int n_elements = strides[0] * tdims[0];
    std::valarray<std::size_t> sizes(tdims.data(), tdims.size());
    sizes[dim] = 1;
    std::valarray<std::size_t> id_data(n_elements);
    std::iota(std::begin(id_data), std::end(id_data), 0);
    const std::valarray<std::size_t> idxs = id_data[std::gslice(0, sizes, strides)];

    return {strides, idxs};
}

// https://en.cppreference.com/w/cpp/locale/numpunct/truefalsename
struct custom_tf : std::numpunct<char>
{
    std::string do_truename() const { return {"True"}; }
    std::string do_falsename() const { return {"False"}; }
};

/**
 * @brief create a formatted string rep of an ND vector with the given dims
 *
 * @param nd_data(type std::valarray<float>)
 * @param shape(type std::vector<int>)
 *
 * @return stringstream object(type std::stringstream)
 */
template <class T>
std::stringstream printND(std::valarray<T> *nd_data, std::vector<std::size_t> shape)
{
    int n_elements = nd_data->size();
    const bool isBool = *typeid(T).name() == 'b';
    std::stringstream out;
    out.setf(std::numeric_limits<T>::digits10);
    int MAX_WIDTH;
    if (isBool)
        MAX_WIDTH = 4;
    else
        MAX_WIDTH = std::to_string(nd_data->max()).length();
    std::string dl(shape.size(), '[');
    out << dl;
    auto [strides, idxs] = generate_idxs(shape, shape.size() - 1);
    strides[shape.size() - 1] = strides[0] * shape[0];
    for (auto i = 0; auto idx : idxs)
    {
        if (i != n_elements && i != 0)
        {
            auto count = std::count_if(std::begin(strides), std::end(strides), [=](int s)
                                       { return idx % s == 0; });
            std::string ddl(count, ']'), sp(count, '\n'), dfl(count, '[');
            out << ddl << "," << sp;
            out.width(shape.size() + 1);
            out << dfl;
        }
        std::valarray<T> sl = (*nd_data)[std::slice(idx, shape[shape.size() - 1], 1)];
        if (isBool)
        {
            out.setf(std::ios_base::boolalpha);
            out.setf(std::ios_base::skipws);
            out.imbue(std::locale(std::cout.getloc(), new custom_tf));
        }
        out << std::setprecision(4);
        out.width(MAX_WIDTH-1); out<<std::right;
        out << sl[0];
        std::for_each(std::begin(sl) + 1, std::end(sl), [&](T n)
                      { 
            out<<std::left<<",";
            out<<std::setprecision(4);
            out.width(MAX_WIDTH+1);
            out<<std::right;
            out<<n; });
        out.unsetf(std::ios_base::left);
        out.unsetf(std::ios_base::right);
        out.unsetf(std::ios_base::skipws);
        out.width(0);
        out << " ";
        i++;
    }
    std::string ddl(shape.size(), ']');
    out << ddl;

    return out;
};

template <class T>
void CHECK_BACKWARD(const std::vector<std::shared_ptr<T>> &var, int expected = 1)
{
    std::string err_msg = "cant backprop without executing a forward computation first";
    assertm(var.size() != expected, err_msg);
    if (var.size() != expected)
        throw std::runtime_error(err_msg);
};
/**
 *
 */
inline void CHECK_TRANSPOSE(std::vector<size_t> s, int a, int b)
{
    const bool isValid = (-s.size() <= a < s.size()) && (-s.size() <= b < s.size()) && (std::abs(a - b) == 1);
    assertm(isValid, ERROR_TRANSPOSE);
    if (!isValid)
        throw std::runtime_error(ERROR_TRANSPOSE);
}
/**
 *
 * @brief to check inputs to an inplace binary operator
 *
 * @param lhs(type std::shared_ptr<T>)
 */
template <class T>
void CHECK_ARGS_IN_PLACE_OPS(const std::shared_ptr<T> &lhs)
{
    assertm(lhs->requires_grad == false, ERROR_IN_PLACE_OP_LEAF);
    if (lhs->requires_grad())
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

// https://pytorch.org/docs/stable/notes/broadcasting.html  check if tensors are "broadcastable"
inline void CHECK_ARGS_OPS_BROADCAST(const std::vector<size_t> dims, const std::vector<size_t> tdims)
{
    for (int i = -1; i >= -std::min(dims.size(), tdims.size()); i--)
    {
        if (std::min(dims[dims.size() + i], tdims[tdims.size() + i]) != 1 && tdims[tdims.size() + i] != dims[dims.size() + i])
            throw std::runtime_error(ERROR_SIZE_MISMATCH);
    }
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
 * @brief check valid dims matmul operation
 *  r_tensor = AxBxCx..xDxE
 *  l_tensor = AxBxCx..xExD
 *  r_tensor mm l_tensor = AxBxCx...xDxD
 *
 * @param dims(type std::vector<int>)
 * @param tdims(type std::vector<int>)
 */
inline void CHECK_MM_DIMS(std::vector<size_t> ldims, std::vector<size_t> rdims)
{
    std::iter_swap(ldims.rbegin(), ldims.rbegin() + 1);
    int min_d = std::min(rdims.size(), ldims.size());
    bool isValid = std::equal(rdims.rbegin() + 1, rdims.rbegin() + min_d, ldims.rbegin() + 1, ldims.rbegin() + min_d);
    assertm(isValid, ERROR_MM_COMPATIBLE);
    if (!isValid)
        throw std::runtime_error(ERROR_MM_COMPATIBLE);
}

/**
 * @brief check valid dim given the rank of a tensor
 *
 * @param dims(type std::vector<int>)
 * @param tdims(type std::vector<int>)
 */
inline void CHECK_VALID_RANGE(const int &dim, const int &rank, const int &low = 0)
{
    assertm(dim == INT_MAX || low <= dim < rank, ERROR_OUT_OF_BOUND_DIM);
    if (dim != INT_MAX && (dim >= rank || dim < low))
        throw std::runtime_error(ERROR_OUT_OF_BOUND_DIM);
}

inline void CHECK_EQUAL_SIZES(const std::vector<size_t> dims1, const std::vector<size_t> dims2)
{
    const bool isValid = dims1.size() == dims2.size() && std::equal(dims1.begin(), dims1.end(), dims2.begin());
    if (!isValid)
        throw std::runtime_error("invalid op, tensors sizes must be the same");
}

template <class T>
std::valarray<T> *repeat_nd(std::valarray<T> *d, const std::vector<size_t> &dims, const std::unordered_map<int, std::size_t> n_repeat)
{
    int n_elements = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    auto out_data = new std::valarray<T>(n_elements);

    for (const auto &[dim, n] : n_repeat)
    {
        const auto &[strides, idxs] = generate_idxs(dims, dim);
        for (int i = 0; const auto &id : idxs)
            (*out_data)[std::slice(id, n, strides[dim])] = (*d)[i++ % d->size()];
    }

    return out_data;
}

bool is_broadcastable(const std::vector<size_t> &dims, const std::vector<size_t> &tdims)
{
    for (int i = -1; i >= -std::min(dims.size(), tdims.size()); i--)
    {
        if (std::min(dims[dims.size() + i], tdims[tdims.size() + i]) != 1 && tdims[tdims.size() + i] != dims[dims.size() + i])
            return false;
    }
    return true;
}

// https://numpy.org/doc/stable/user/basics.broadcasting.html
// https://pytorch.org/docs/stable/notes/broadcasting.html
template <class T>
void broadcast(std::valarray<T> *lhs, std::vector<size_t> lhs_dims, std::valarray<T> *rhs, std::vector<size_t> rhs_dims, std::vector<size_t> *new_dims)
{
    int n_dims = std::max(rhs_dims.size(), lhs_dims.size());
    std::unordered_map<int, size_t> n_repeat_lhs, n_repeat_rhs;

    int i = -1;

    while (i >= -n_dims)
    {
        size_t s, l, r;
        int idx = n_dims + i;
        if (std::abs(i) > rhs_dims.size() && idx < lhs_dims.size())
        { // second condition not neccessary since rhs_dims<>lhs_dims
            s = lhs_dims[idx];
            n_repeat_rhs[idx] = s;
            rhs_dims.insert(rhs_dims.begin(), s);
        }
        else if (std::abs(i) > lhs_dims.size())
        {
            s = rhs_dims[idx];
            n_repeat_lhs[idx] = s;
            lhs_dims.insert(lhs_dims.begin(), s);
        }
        else
        {
            r = rhs_dims.size() + i;
            l = lhs_dims.size() + i;
            s = std::max(lhs_dims[l], rhs_dims[r]);
            if (lhs_dims[l] != rhs_dims[r])
                lhs_dims[l] > rhs_dims[r] ? n_repeat_rhs[r] = s : n_repeat_lhs[l] = s;
            lhs_dims[l] = s;
            rhs_dims[r] = s;
        }
        i--;
    }
    *new_dims = rhs_dims;

    if (!n_repeat_lhs.empty())
    {
        auto temp_lhs = repeat_nd(lhs, lhs_dims, n_repeat_lhs);
        *lhs = *temp_lhs;
    }
    if (!n_repeat_rhs.empty())
    {
        auto temp_rhs = repeat_nd(rhs, rhs_dims, n_repeat_rhs);
        *rhs = *temp_rhs;
    }
}
#endif