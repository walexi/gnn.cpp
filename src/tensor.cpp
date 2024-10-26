#include "operation.h"
#include "tensor.h"
#include <memory>
#include <iostream>
#include <random>
#include <numeric>
#include <algorithm>
#include <format>
#include <string>
#define NDEBUG
#include <cassert>

using namespace cyg;
using namespace std;


int flogRefCount = 1; //use to log tensors ref count, 0=yes 1= no logging
default_random_engine e(time(nullptr)); 

/**
 * @brief use to create a vector of input dims and value,
 * 
 * @param dims(type std::vector<int>) //@todo array
 * @param value(type int)
 * 
 * @return pointer to vector of floats(type std::vector<float>)
 */
vector<float> *cyg::initialize(vector<int> &dims, int value)
{
    auto n_elements = accumulate(dims.begin(), dims.end(), 1, multiplies<float>());
    auto vec = new vector<float>(n_elements, value);
    return vec;
}
/**
 * @brief constructor for the tensor class
 * 
 * @param dims(type std::vector<int>)
 * @param device(type cyg::Device)
 * @param requires_grad(type bool)
 */
cyg::tensor::tensor(std::vector<int> &dims, Device device, bool requires_grad) : dims(dims), device(device), requires_grad(requires_grad)
{   
    assertm(dims.size() != 0, "dims cannot be empty");
    assertm(dims.min() > 0, "dim cannot be zero");
    if (*min_element(dims.begin(), dims.end()) < 1 || dims.size() == 0)
        throw runtime_error("dims array cannot be empty or less than 1");
    this->d = initialize(dims, 0);
    if (requires_grad)
        this->zero_grad();
}
/**
 * @brief constructor for the tensor class
 * 
 * @param data(type const std::vector<float>)
 * @param dims(type std::vector<int>)
 * @param device(type cyg::Device)
 * @param requires_grad(type bool)
 */
cyg::tensor::tensor(const std::vector<float> &data, std::vector<int> &dims, Device device, bool requires_grad) : d(&data), dims(dims), device(device), requires_grad(requires_grad)
{
    // if(N==0) throw runtime_error("dims array cannot be empty");
    assertm(*min_element(dims.begin(), dims.end()) > 0, "dim cannot be zero");
    size_t n_elements = accumulate(dims.cbegin(), dims.cend(), 1, multiplies<size_t>());
    // static_assert(n_elements==data->size());
    if (n_elements != data.size())
        throw runtime_error("mismatch between data size and dims");
    if (requires_grad)
        this->zero_grad();
};

/**
 * @brief class method to set the grad of the tensor to zero
 * 
 */
void cyg::tensor::zero_grad()
{
    this->grad = initialize(this->dims, 0);
};
/**
 * @brief class method to move tensor from gpu to cpu
 */
void cyg::tensor::detach() {
    // move_tensor(this, Device::cpu);
};
/**
 * @brief class method to turn on/off grad for the given tensor
 * 
 * @param requires_grad(type bool)
 */
void cyg::tensor::enable_grad(bool requires_grad)
{
    if (requires_grad)
    {
        this->requires_grad = true;
        this->zero_grad();
    }
    else
    {
        this->requires_grad = false;
        this->grad = nullptr;
    };
};
/**
 * @brief remove dim with size of 1 from the give tensor
 * 
 */
void cyg::tensor::squeeze()
{
    auto it = find(this->dims.begin(), this->dims.end(), 1);
    while (it != this->dims.end())
    {
        this->dims.erase(it);
        it = find(it + 1, this->dims.end(), 1);
    };
};
/**
 * @brief add dim of size 1 at the input dim of the given tensor
 * 
 * @param dim(type int)
 */
tensor &cyg::tensor::unsqueeze(int dim)
{
    bool isvalid = -this->rank() - 1 <= dim <= this->rank();
    // string message = std::vformat("dim must be within range of [-{0}-1, {0}]", std::make_format_args(this->rank()));
    string message = std::format(std::string_view("dim must be within range of [-{0}-1, {0}]"), this->rank());
    if (!isvalid)
    {
        // assert(message);
        throw runtime_error(message);
    }
    this->dims.insert(this->dims.begin() + (dim < 0 ? dim + this->rank() + 1 : dim), 1);
    return *this;
}
/**
 * @brief calculate the index of an element in a flatten array given the ND dims/cordinates
 *                                                             t_dims = {1,5,7}
 *  dims = {1,5,7} n_elements=1*5*7=35 last_index in the flatten vec is {0,4,6} = 0*5 + 4*7 + 6=12
 *                                                              dims = {0,0,1} = 0*2 + 0*3 + 1 = 1
 *
 * @param t_dims(type std::valarray<int>) N_D dims of the tensor
 * @param dims(type std::valarray<int>) N_D dims of the element
 *
 * @return the 1D index of the element (type int)
 * */
auto get_index(vector<int> t_dims, vector<int> dims)
{
    transform(dims.cbegin(), dims.cend(), next(t_dims.cbegin()), dims.begin(), multiplies<int>());
    auto index = accumulate(dims.cbegin(), dims.cend(), dims[-1]);
    return index;
};
/**
 * @brief class method to backprop on given tensor
 * 
 * @param incoming_gradient(type std::vector<float>)
 */
void cyg::tensor::backward(std::vector<float>* incoming_gradient)
{
    // assertm(this->requires_grad, "please enable requires_grad on tensor to backprop");
    if (incoming_gradient == nullptr && this->n_elements()!=1) throw runtime_error("pass in tensor to backprop on non-scalar tensor");
    if(this->n_elements()==1 && incoming_gradient==nullptr) incoming_gradient = initialize(this->dims, 1);
    if(this->grad!=nullptr){
        *this->grad = *this->grad + *incoming_gradient;
    }
    
    if(this->grad_fn!=nullptr) this->grad_fn->backward(incoming_gradient);

};
/**
 * @brief operator overloading for the function call. 
 * Use to get an element from the tensor
 * 
 * @return element at the input elemet (type float)
 */
float tensor::operator()(int d, ...)
{
    vector<int> dims = {d};
    va_list args;
    va_start(args, d);
    for (int i = 0; i < this->rank() - 1; ++i)
        dims.push_back(va_arg(args, int));
    va_end(args);
    for (int i = 0; i < this->rank(); ++i)
        if (dims[i] >= this->shape()[i] || dims[i] < 0)
            throw runtime_error("out of bound range");
    auto index = get_index(this->dims, dims);
    return (*this->d)[index];
}
/**
 * @brief set elements of a 2D tensor below the main diagonal to zero, i.e a_i,j = 0 for
 *
 *         [1,2,3,4,5,6,7,8,9], 3*3 =
 *          [
 *             [1,2,3],
 *             [4,5,6],
 *             [7,8,9]
 *          ]
 * triu =
 *          [
 *             [1,2,3], 1,2,3,(4),5,6,(7),(8),9
 *            [0,5,6]
 *            [0,0,9]
 *          ]
 *          [
 *            [1,2,3,4], 1,2,3,4,(5),6,7,8,(9),(10),11,12,(13),(14),(15),16
 *            [0,6,7,8]
 *            [0,0,11,12]
 *            [0,0,0,16]
 *          ]
 * for i=D-1,i<N,i+D
 *      arr[i:i-D+1]=0
 *
 * 4*4, D=4, N=16
 * i=D = 4
 * while(i<N)  i=4,8
 *  arr[i: i-D]=0 //arr[4: 4-4]=arr[4: 4-4 + D - 16//4 +1], arr[8: 8-4 + 16//8 -1]
 *  i+=D
 * i=4 i - N//i + 1 = D - 16//4 = 4 - 4 = 0
 * i=8              = D - 16//8 = 4 - 2 = 2
 * i=12             = D - 16//12 = 4 - 1 = 3
 *
 * i=3    = 3 - 9//3 = 3 - 3 = 0
 * i=6    = 3 - 9//6 = 3 - 1 = 2
 */
std::shared_ptr<tensor> cyg::tensor::triu()
{
    return std::shared_ptr<tensor>();
}
/**
 * @brief addition operation for tensors, tensors must have equal shape
 * 
 * @param other(type: cyg::tensor)
 * @return tensor (type cyg::tensor)
 */
std::shared_ptr<tensor> cyg::tensor::add(const std::shared_ptr<tensor> &other)
{
    // @todo move repeated assert/checks to util
    assertm(this->n_elements() ==other->n_elements(), "tensors must have shape"); //just here to fail fast/quick - redundant though, same assert in Add operator
    if(this->n_elements()!=other->n_elements()) throw runtime_error("tensors are of different shapes");
    auto add_op = new Add<tensor>();
    auto output = add_op->forward(shared_from_this(), other);
    output->grad_fn = add_op;
    return output;
};

/**
 * @brief element wise multiplication operation for tensors, tensors must have equal shape
 * 
 * @param other(type: cyg::tensor)
 * @return tensor (type cyg::tensor)
 */
std::shared_ptr<tensor> cyg::tensor::mul(const std::shared_ptr<tensor> &other)
{
    assertm(this->n_elements() ==other->n_elements(), "tensors must have shape"); //just here to fail fast/quick - redundant though, same assert in Add operator
    if(this->n_elements()!=other->n_elements()) throw runtime_error("tensors are of different shapes");
    auto mul_op = new Mul<tensor>();
    auto output = mul_op->forward(shared_from_this(), other);
    output->grad_fn = mul_op;
    return output;
};

/**
 * @brief scalar multiplication operation for tensors, multiply a tensor by a scalar
 * 
 * @param other(type: float)
 * @return tensor (type cyg::tensor)
 */
std::shared_ptr<tensor> cyg::tensor::operator*(const float &rhs)
{
    return this->mul(fill(rhs,this->dims, this->device, this->requires_grad));
}

/**
 * @brief element wise division operation for tensors, tensors must have equal shape
 * 
 * @param other(type: cyg::tensor)
 * @return tensor (type cyg::tensor)
 */
std::shared_ptr<tensor> cyg::tensor::div(const std::shared_ptr<tensor> &other)
{
    assertm(this->n_elements() ==other->n_elements(), "tensors must have shape"); //just here to fail fast/quick - redundant though, same assert in Add operator
    if(this->n_elements()!=other->n_elements()) throw runtime_error("tensors are of different shapes");
    auto div_op = new Div<tensor>();
    auto output = div_op->forward(shared_from_this(), other);
    output->grad_fn = div_op;
    return output;
};
/**
 * @brief element wise exponent operation for tensors, tensors must have equal shape
 * 
 * @param other(type: cyg::tensor)
 * @return tensor (type cyg::tensor)
 */
std::shared_ptr<tensor> cyg::tensor::pow(const std::shared_ptr<tensor>& exponent)
{
    assertm(this->n_elements() ==exponent->n_elements(), "tensors must have shape"); //just here to fail fast/quick - redundant though, same assert in Add operator
    if(this->n_elements()!=exponent->n_elements()) throw runtime_error("tensors are of different shapes");
    auto pow_op = new Pow<tensor>();
    auto output = pow_op->forward(shared_from_this(), exponent);
    output->grad_fn = pow_op;
    return output;
}

/**
 * @brief scalar exponent operation for tensors, raise a tensor to a scalar power
 * 
 * @param other(type: cyg::tensor)
 * @return tensor (type cyg::tensor)
 */
std::shared_ptr<tensor> cyg::tensor::pow(const float& exponent)
{ 
    const auto t = fill(exponent, this->dims, this->device, this->requires_grad);
    auto out = this->pow(t);
    return out;
};
/**
 * @brief compute the mean along the input dimension for a given tensor
 * for ex
 *  auto t = cyg::ones({2,3,4})
 *  t.mean(2) // mean along the dimension of size 4
 * 
 * @param dims(type: int)
 * @param keepdims(type: bool)
 * 
 * @return tensor (type cyg::tensor)
 */
std::shared_ptr<tensor> cyg::tensor::mean(const int dims, const bool keepdims)
{ 
    string err_msg = std::format(std::string_view("index out of range, it should be in the range [0, {}]"), this->rank());
    assertm(0<=dims<this->rank(), err_msg);
    if(dims<0 || dims>=this->rank()) throw runtime_error(err_msg);
    auto mean_op = new Mean<tensor>();
    auto output = mean_op->forward(shared_from_this(), dims, keepdims);
    output->grad_fn = mean_op;
    return output;
}

/**
 * @brief compute the exponent of a tensor along
 * for ex
 * 
 * 
 * @return tensor (type std::shared_ptr<cyg::tensor>)
 */
std::shared_ptr<tensor> cyg::tensor::exp()
{ 
    auto exp_op = new Exp<tensor>();
    auto output = exp_op->forward(shared_from_this());
    output->grad_fn = exp_op;
    return output;
}


/**
 * @brief compute the natural log of a tensor along
 * for ex
 * 
 * 
 * @return tensor (type std::shared_ptr<cyg::tensor>)
 */
std::shared_ptr<tensor> cyg::tensor::log()
{ 
    auto log_op = new Log<tensor>();
    auto output = log_op->forward(shared_from_this());
    output->grad_fn = log_op;
    return output;
}

/**
 * @brief generate a floating-point number from a uniform dist in the range [low, high)
 * pdf = 1/(high-low)
 * 
 * @param low(type int)
 * @param high(type int)
 * 
 * @return random number(type float)
 */
float generate_random(int low, int high)
{
    uniform_real_distribution<float> u(low, high);
    return u(e);
}

/**
 * @brief create a tensor with randomly generated data from a uniform distribution
 * @ todo add a param for a generator (different distributions)
 * 
 * @param dims(type std::vector<int>)
 * @param low(type int)
 * @param high(type int)
 * @param device(type cyg::Device)
 * @param requires_grad(type bool)
 * 
 * @return generated tensor(type std::shared_ptr<tensor>)
 */
std::shared_ptr<tensor> cyg::randn(std::vector<int> &dims, int low, int high, Device device, bool requires_grad)
{
    assertm(low<high, "low must be lower than high, pls check your input params");
    if(low>=high) throw runtime_error("pls check input params, the value for the low arg must be lower than the high args");
    auto vec = initialize(dims, 1);
    generate(vec->begin(), vec->end(), [low, high]()
             { return generate_random(low, high); });
    return make_shared<tensor>(*vec, dims, device, requires_grad);
}

/**
 * @brief create a tensor with the given input constant
 * 
 * 
 * @param value(type float)
 * @param dims(type std::vector<int>)
 * @param device(type cyg::Device)
 * @param requires_grad(type bool)
 * 
 * @return generated tensor(type std::shared_ptr<tensor>)
 */
std::shared_ptr<tensor> cyg::fill(const float& value, std::vector<int> &dims, Device device, bool requires_grad)
{
    auto vec = initialize(dims, value);
    return make_shared<tensor>(*vec, dims, device, requires_grad);
}

/**
 * @brief create a tensor with 1s data
 * 
 * 
 * @param dims(type std::vector<int>)
 * @param device(type cyg::Device)
 * @param requires_grad(type bool)
 * 
 * @return generated tensor(type std::shared_ptr<tensor>)
 */
std::shared_ptr<tensor> cyg::ones(std::vector<int> &dims, Device device, bool requires_grad)
{
    return fill(1, dims, device, requires_grad);
}

/**
 * @brief create a tensor with 0s data
 * 
 * 
 * @param dims(type std::vector<int>)
 * @param device(type cyg::Device)
 * @param requires_grad(type bool)
 * 
 * @return generated tensor(type std::shared_ptr<tensor>)
 */
std::shared_ptr<tensor> cyg::zeros(std::vector<int> &dims, Device device, bool requires_grad)
{
    return fill(0, dims, device, requires_grad);
}
/**
 * @brief create a tensor with 1s and same shape as input tensor
 * 
 * 
 * @param input_tensor(type std::shared_ptr<tensor>)
 * @param device(type cyg::Device)
 * @param requires_grad(type bool)
 * 
 * @return generated tensor(type std::shared_ptr<tensor>)
 */
std::shared_ptr<tensor> cyg::ones_like(const std::shared_ptr<tensor> &input_tensor, Device device, bool requires_grad)
{
    return ones(input_tensor->shape(), device, requires_grad);
}

/**
 * @brief create a tensor with 0s and same shape as input tensor
 * 
 * 
 * @param input_tensor(type std::shared_ptr<tensor>)
 * @param device(type cyg::Device)
 * @param requires_grad(type bool)
 * 
 * @return generated tensor(type std::shared_ptr<tensor>)
 */
std::shared_ptr<tensor> cyg::zeros_like(const std::shared_ptr<tensor> &input_tensor, Device device, bool requires_grad)
{
    return zeros(input_tensor->shape(), device, requires_grad);
};

/**
 * @brief create a formatted string rep of an ND vector with the given dims
 * 
 * @param nd_data(type std::vector<float>)
 * @param shape(type std::vector<int>)
 * 
 * @return stringstream object(type std::stringstream)
 */
stringstream cyg::printND(vector<float> nd_data, vector<int> shape)
{
    int n_elements = nd_data.size();
    stringstream out;
    // out.setf(std::numeric_limits<float>::digits10);
    string dl(shape.size(), '[');
    out << dl << " ";
    int i = 0;
    while (i < n_elements)
    {
        out << nd_data[i];
        int count = 0;
        int j = shape.size() - 1;
        int div = shape[j];
        if (i + 1 != n_elements)
        {
            while (j > 0)
            {
                if ((i + 1) % div == 0) count++;
                j--;
                div = div * shape[j];
            }
            if (count > 0)
            {
                string ddl(count, ']');
                string sp(count, '\n');
                string dfl(count, '[');
                out << ddl << "," << sp << dfl;
            }
            else out << " , ";
        }
        i++;
    }
    string el(shape.size(), ']');
    out << el;
    return out;
}


/**
 * @brief overloading cout operator for tensor
 * 
 * @param out(type std::ostream)
 * @param input_tensor(type cyg::tensor)
 * 
 * @return output_stream(type std::ostream)
 */
std::ostream &cyg::operator<<(std::ostream &out, const tensor &t)
{
    auto data = t.data();
    auto shape = t.shape();
    out << "(";
    stringstream output_string = printND(*data, shape);
    out << output_string.str();
    out << ", size = ( ";
    for (auto r : t.shape())
        out << r << " ";
    out << "), requires_grad = ";
    string require_grad = (t.require_grad() == 0) ? "false" : "true";
    string device = (t.get_device() == Device::cpu) ? "cpu" : "cuda";
    out << require_grad << ", device = " << device << " ]" << endl;
    return out;
}