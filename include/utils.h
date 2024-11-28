#ifndef UTIL_H
#define UTIL_H
#include <valarray>
#include <vector>
#include <unordered_map>
#include <numeric>
#include <assert.h>
#include <algorithm>
#include <random>
#include <iomanip>
#include <sstream>
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


void CHECK_TRANSPOSE(std::vector<size_t> s, int a, int b);

/**
 * @brief check valid dims
 *
 * @param dims(type std::vector<int>)
 */
void CHECK_VALID_DIMS(std::vector<size_t> dims);

/**
 * @brief check rank of given dims
 *
 * @param dims(type std::vector<int>)
 * @param tdims(type std::vector<int>)
 */
void CHECK_RANK(std::vector<size_t> dims, std::vector<size_t> tdims);

/**
 * @brief check tensor's data size given input dims
 *
 * @param dims(type std::vector<int>)
 * @param n_elements(type int)
 */
void CHECK_SIZE(std::vector<size_t> dims, int n_elements);

// https://pytorch.org/docs/stable/notes/broadcasting.html  check if tensors are "broadcastable"
void CHECK_ARGS_OPS_BROADCAST(const std::vector<size_t> dims, const std::vector<size_t> tdims);

/**
 * @brief check valid dims for indexing a tensor given the tensor's dims
 *
 * @param dims(type std::vector<int>)
 * @param tdims(type std::vector<int>)
 */
void CHECK_VALID_INDEX(std::vector<size_t> dims, std::vector<size_t> tdims);
/**
 * @brief check valid dims matmul operation
 *  r_tensor = AxBxCx..xDxE
 *  l_tensor = AxBxCx..xExD
 *  r_tensor mm l_tensor = AxBxCx...xDxD
 *
 * @param dims(type std::vector<int>)
 * @param tdims(type std::vector<int>)
 */
void CHECK_MM_DIMS(std::vector<size_t> ldims, std::vector<size_t> rdims);
/**
 * @brief check valid dim given the rank of a tensor
 *
 * @param dims(type std::vector<int>)
 * @param tdims(type std::vector<int>)
 */
void CHECK_VALID_RANGE(const int &dim, const int &rank, const int &low = 0);
void CHECK_EQUAL_SIZES(const std::vector<size_t> dims1, const std::vector<size_t> dims2);
/**
 * @brief generate a floating-point number from a uniform dist in the range [low, high)
 * pdf = 1/(high-low)
 *
 * @param low(type int)
 * @param high(type int)
 *
 * @return random number(type float)
 */
float generate_random(const float &low, const float &high);

/**
 * @brief calculate the index of an element in a flatten array given the ND dims/cordinates
 *
 * @param t_dims(type std::valarray<int>) N_D dims of the tensor
 * @param dims(type std::valarray<int>) N_D dims of the element
 *
 * @return the 1D index of the element (type int)
 * */
int get_index(std::vector<std::size_t>* t_dims, std::vector<std::size_t> dims);

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
 * ids across the col with dim of size of 4 has 5 elements are = 1, 5, 9, 13, 17
 *
 * ids across the row with dim of size of 5 has 4 elements are = 1, 2, 3, 4
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
 * ids across the col with dim of size 4, has 2*5 elements and are = 1, 5, 9, 13, 17, 21, 25, 29, 33, 37
 * ids across the row with dim of size 5, has 2*4 elements and are = 1, 2, 3, 4, 21, 22, 23, 24
 * ids across the batch with dim of size 2, has 5*4 elements and are =  1,2,3,4,5,6,7,8,9,....,19,20
 */

std::tuple<std::valarray<std::size_t>, std::valarray<std::size_t>> generate_idxs(const std::vector<std::size_t> tdims, int dim);

bool is_broadcastable(const std::vector<size_t> &dims, const std::vector<size_t> &tdims);

/**
 * @brief use to create am array with the input dims and value,
 *
 * @param dims(type std::array<int, N>) //@todo array
 * @param value(type int)
 *
 * @return unique_ptr of to an array of type T(type std::unique_ptr<std::valarray<T>>)
 */
template <class T>
std::valarray<T>* initialize(std::vector<size_t> dims, T value = 1)
{
    auto n_elements = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
    auto data = new std::valarray<T>(n_elements);
    std::fill_n(std::begin(*data), n_elements, value);

    return data;
};

template <class T>
std::valarray<T> *repeat_nd(std::valarray<T> *d, const std::vector<size_t> &dims, const std::unordered_map<int, std::size_t> n_repeat)
{
    int n_elements = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    auto out_data = new std::valarray<T>(n_elements);

    for (const auto &[dim, n] : n_repeat)
    {
        const auto &[strides, idxs] = generate_idxs(dims, dim);
        for (int i = 0; const auto &id : idxs){
            (*out_data)[std::slice(id, n, strides[dim])] = (*d)[i++ % d->size()];
        }
    }

    return out_data;
};

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
};

// https://en.cppreference.com/w/cpp/locale/numpunct/truefalsename
struct custom_tf : std::numpunct<char>
{
    std::string do_truename() const { return {"True"}; }
    std::string do_falsename() const { return {"False"}; }
};

/**
 * @brief create a formatted string rep of an ND array with the given dims
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
            out.imbue(std::locale(out.getloc(), new custom_tf));
        }
        out << std::setprecision(4);
        out.width(MAX_WIDTH-1); out<<std::right;
        out << sl[0];
        std::for_each(std::begin(sl) + 1, std::end(sl), [&](T n)
                      { 
            out<<std::left<<",";
            out<<std::setprecision(4);
            out.width(MAX_WIDTH+1);
            out<<std::right<<n; });
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

template <class T>
void CHECK_BACKWARD(const std::vector<std::shared_ptr<T>> &var, int expected = 1)
{
    std::string err_msg = "cant backprop without executing a forward computation first";
    assertm(var.size() != expected, err_msg);
    if (var.size() != expected)
        throw std::runtime_error(err_msg);
};
#endif