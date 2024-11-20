#ifndef FUNCTIONAL_H
#define FUNCTIONAL_H

#include "utils.h"
#include <unordered_map>
#include <memory>
#include <valarray>

namespace cyg 
{
    template<class T>class tensor;

}
/**
 * operations without backprop
 * 
 * always clone tensors input
 * use tensors with req_grad set to false when you have to execute ops on tensors
 */
namespace functional
{
    /**
     * return the indices and values of the max values in the input tensor
     * the max val can be computed along a dimension or over all the elements (if no dim is specified)
     */
    template<class T>
    std::tuple<std::shared_ptr<cyg::tensor<T>>, std::shared_ptr<cyg::tensor<int>>> max(const std::shared_ptr<cyg::tensor<T>>& input_tensor, int dim, const bool& keepdim)
    {

        std::vector<size_t> new_dims;
        
        std::valarray<int> id_data;

        auto input_data = (*input_tensor->data());

        auto max_values = new std::valarray<T>(1);
        auto max_idxs = new std::valarray<int>(1);

        if (dim == INT_MAX) // flattened input
        {
            id_data.resize(input_tensor->numel());
            std::iota(std::begin(id_data), std::end(id_data), 0);
            auto max_val =  input_data.max();
            (*max_values)[0] = max_val;
            (*max_idxs)[0] = std::valarray(id_data[input_data == max_val])[0];
            new_dims = {1};
        }
        else
        {
            if (dim < 0) dim = input_tensor->rank() + dim;
            id_data.resize(input_tensor->shape()[dim]);
            std::iota(std::begin(id_data), std::end(id_data), 0);

            const auto [strides, start_idxs] = generate_idxs(input_tensor->shape(), dim);

            max_values->resize(start_idxs.size());
            max_idxs->resize(start_idxs.size());

            for (auto i = 0; const auto &idx : start_idxs)
            {
                auto gslice = std::slice(idx, input_tensor->shape()[dim], strides[dim]);
                auto data_slice = std::valarray(input_data[gslice]);
                auto max_val = data_slice.max();
                (*max_values)[i] = max_val;
                (*max_idxs)[i++] = std::valarray(id_data[data_slice == max_val])[0];
            }
            new_dims = input_tensor->shape();
            new_dims[dim] = 1;
            if (!keepdim)
                new_dims.erase(new_dims.begin() + dim);
        }
        auto value_tensor = std::make_shared<cyg::tensor<T>>(new_dims, max_values, false, false);
        auto indices_tensor = std::make_shared<cyg::tensor<int>>(new_dims, max_idxs, false, false);

        return {value_tensor, indices_tensor};
    }


    /**
     * computes elements wise maximum of the given tensors
     * tensors sizes must be the same
     */
    template<class T>
    std::shared_ptr<cyg::tensor<T>> maximum(const std::shared_ptr<cyg::tensor<T>>& tensor1, const std::shared_ptr<cyg::tensor<T>>& tensor2){

        CHECK_EQUAL_SIZES(tensor1->shape(), tensor2->shape());
        int n_elements = tensor1->numel();

        auto out_data = new std::valarray<T>(n_elements);
        auto t1data = (*tensor1->data());
        auto t2data = (*tensor2->data());

        for(int i=0; i<n_elements; i++){
            (*out_data)[i] = std::max(t1data[i], t2data[i]);
        }

        return std::make_shared<cyg::tensor<T>>(tensor1->shape(), out_data, false, false);
    }

    /**
     * computes the maximum of the tensor's elements  and the given value
     */
    template<class T>
    std::shared_ptr<cyg::tensor<T>> maximum(const std::shared_ptr<cyg::tensor<T>>& tensor1, const T value){

        return maximum<T>(tensor1, std::make_shared<cyg::tensor<T>>(tensor1->shape(), static_cast<T>(value), false, false));
    }

    template<class T>
    std::shared_ptr<cyg::tensor<T>> abs(const std::shared_ptr<cyg::tensor<T>>& t)
    {
        auto cloned_t = t->clone();
        auto new_data = new std::valarray<T>();
        *new_data = std::abs(*t->data());
        cloned_t->set_data(new_data);
    
        return cloned_t;
    }
     /**
     * computes elements wise maximum of the given tensors
     * tensors sizes must be the same
     */
    template<class T>
    std::shared_ptr<cyg::tensor<T>> minimum(const std::shared_ptr<cyg::tensor<T>>& t1, const std::shared_ptr<cyg::tensor<T>>& t2){

        CHECK_EQUAL_SIZES(t1->shape(), t2->shape());
        auto tensor1 =  t1->clone();
        auto tensor2 = t2->clone();
        return -(maximum<T>(-tensor1, -tensor2));
    }

    /**
     * computes the minimum of the tensor's elements  and the given value
     */
    template<class T>
    std::shared_ptr<cyg::tensor<T>> minimum(const std::shared_ptr<cyg::tensor<T>>& tensor1, const T value){

        return minimum<T>(tensor1, std::make_shared<cyg::tensor<T>>(tensor1->shape(), static_cast<T>(value), false, false));
    }

    template<class T>
    std::shared_ptr<cyg::tensor<bool>> gt(const std::shared_ptr<cyg::tensor<T>>& t1, const std::shared_ptr<cyg::tensor<T>>& t2)
    {
        auto t1data = new std::valarray<T>(), t2data = new std::valarray<T>();
        auto new_dims = new std::vector<size_t>();
        *t1data = *t1->data();
        *t2data = *t2->data();
        if(is_broadcastable(t1->shape(), t2->shape())){
            broadcast(t1data, t1->shape(), t2data, t2->shape(), new_dims);
        }
        auto new_data = new std::valarray<bool>(t1->numel());
        *new_data = *t1data>*t2data;
        delete t1data; delete t2data;

        return std::make_shared<cyg::tensor<bool>>(*new_dims, new_data, false, false);
    }
    
    template<class T>
    std::shared_ptr<cyg::tensor<T>> add(const std::shared_ptr<cyg::tensor<T>>& rhs, const std::shared_ptr<cyg::tensor<T>>& lhs)
    {
        auto req_grad = lhs->requires_grad() || rhs->requires_grad();
        auto out_data = new std::valarray<T>();
        auto new_dims = new std::vector<size_t>(); 
        if(is_broadcastable(lhs->shape(), rhs->shape()))
        {
            auto lhs_data = new std::valarray<T>(); auto rhs_data = new std::valarray<T>();
            *lhs_data = *lhs->data(); *rhs_data = *rhs->data();
            broadcast(lhs_data, lhs->shape(), rhs_data, rhs->shape(), new_dims);
            *out_data = *lhs_data + *rhs_data;
            delete lhs_data; delete rhs_data;
        }else
        {
            *out_data = *lhs->data() + *rhs->data();
            *new_dims = lhs->shape();
        }
        auto output = std::make_shared<cyg::tensor<T>>(*new_dims, out_data, req_grad, false);
        delete new_dims;
        return output;
    }

    template<class T>
    std::shared_ptr<cyg::tensor<T>> mul(const std::shared_ptr<cyg::tensor<T>>& rhs, const std::shared_ptr<cyg::tensor<T>>& lhs)
    {
        auto req_grad = lhs->requires_grad() || rhs->requires_grad();
        auto out_data = new std::valarray<T>();
        auto new_dims = new std::vector<size_t>();
        if(is_broadcastable(lhs->shape(), rhs->shape()))
        {
            auto lhs_data = new std::valarray<T>(); auto rhs_data = new std::valarray<T>();
            *lhs_data = *lhs->data(); *rhs_data = *rhs->data();
            broadcast(lhs_data, lhs->shape(), rhs_data, rhs->shape(), new_dims);
            *out_data = *lhs_data * *rhs_data;
            delete lhs_data; delete rhs_data;
        }else 
        {
            *out_data = *lhs->data() * *rhs->data();
            *new_dims = lhs->shape();
        }
        
        auto output = std::make_shared<cyg::tensor<T>>(*new_dims, out_data, req_grad, false);
        delete new_dims;
        return output;
    }

    template<class T>
    std::shared_ptr<cyg::tensor<T>> div(const std::shared_ptr<cyg::tensor<T>>& numerator, const std::shared_ptr<cyg::tensor<T>>& denominator)
    {
        auto req_grad = numerator->requires_grad() || denominator->requires_grad();
        auto out_data = new std::valarray<T>();
        auto new_dims = new std::vector<size_t>();
        if(is_broadcastable(numerator->shape(), denominator->shape()))
        {
            auto num_data = new std::valarray<T>(); auto den_data = new std::valarray<T>();
            *num_data = *numerator->data(); *den_data = *denominator->data();
            broadcast(num_data, numerator->shape(), den_data, denominator->shape(), new_dims);
            *out_data = *num_data / *den_data;
            delete num_data; delete den_data;
        }else{
            *out_data = *numerator->data() / *denominator->data();
            *new_dims = numerator->shape();
        }
        
        auto output = std::make_shared<cyg::tensor<T>>(*new_dims, out_data, req_grad, false);
        delete new_dims;
        return output;
    }

    template<class T>
    std::shared_ptr<cyg::tensor<T>> pow(const std::shared_ptr<cyg::tensor<T>>& base, const std::shared_ptr<cyg::tensor<T>>& exponent)
    {
        auto req_grad = base->requires_grad() || exponent->requires_grad();
        auto out_data = new std::valarray<T>();
        auto new_dims = std::make_shared<std::vector<size_t>>(); 
        if(is_broadcastable(base->shape(), exponent->shape()))
        {
            auto base_data = new std::valarray<T>(); auto exp_data = new std::valarray<T>();
            *base_data = *base->data(); *exp_data = *exponent->data();
            broadcast(base_data, base->shape(), exp_data, exponent->shape(), new_dims.get());
            *out_data = std::pow(*base_data, *exp_data);
            delete base_data; delete exp_data;
        }else {
            *out_data = std::pow(*base->data(), *exponent->data());
            *new_dims = base->shape();
        }
        
        auto output = std::make_shared<cyg::tensor<T>>( *new_dims, out_data, req_grad, false);

        return output;
    }

    template<class T>
    std::shared_ptr<cyg::tensor<T>> sum(const std::shared_ptr<cyg::tensor<T>> &base, int dim = INT_MAX, const bool& keepdim = false)
    {
        
        dim<0? dim = base->rank() + dim : dim;
        auto data = base->data();
        auto shape = base->shape();
            
        auto out_data = new std::valarray<T>();
        if (dim == INT_MAX)
        {
            out_data->resize(1);
            (*out_data)[0] = data->sum();
            shape = {1};
        }
        else
        {   
            const auto [strides, idxs] = generate_idxs(shape, dim);
            out_data->resize(idxs.size());
            //@todo improve using gslice
            for (int i = 0; i < idxs.size(); ++i)
            {
                (*out_data)[i] = std::valarray((*data)[std::slice(idxs[i], shape[dim], strides[dim])]).sum();
            };
            shape[dim] = 1;
            if (!keepdim) shape.erase(shape.begin() + dim);
        };
        auto output = std::make_shared<cyg::tensor<T>>(shape, out_data, base->requires_grad(), false);

        return output;
    };


    template<class T>
    std::shared_ptr<cyg::tensor<T>> mean(const std::shared_ptr<cyg::tensor<T>> &base, int dim = INT_MAX, const bool& keepdim = false)
    {   
        auto output = sum(base, dim, keepdim);
        dim<0? dim = base->rank() + dim : dim;
        int n_elements;
        dim==INT_MAX? n_elements = base->numel() : n_elements = base->shape()[dim];
        return output/n_elements;
    };

    template<class T>
    std::shared_ptr<cyg::tensor<T>> exp(const std::shared_ptr<cyg::tensor<T>>& base)
    {   
        auto out_data = new std::valarray<T>();
        *out_data = std::exp(*base->data());

        auto output = std::make_shared<cyg::tensor<T>>(base->shape(), out_data, base->requires_grad(), false);

        return output;
    };

    template<class T>
    std::shared_ptr<cyg::tensor<T>> log(const std::shared_ptr<cyg::tensor<T>>& base)
    {   
        auto out_data = new std::valarray<T>();
        *out_data = std::log(*base->data());

        auto output = std::make_shared<cyg::tensor<T>>(base->shape(), out_data, base->requires_grad(), false);

        return output;
    };
    template<class T>
    std::shared_ptr<cyg::tensor<T>> transpose(std::shared_ptr<cyg::tensor<T>> t, int d1, int d2){
        if(d1<0) d1 = t->rank() + d1;
        if(d2<0) d2 = t->rank() + d2;

        auto out_data = new std::valarray<T>(t->numel());

        auto new_dims = t->shape();
        std::iter_swap(new_dims.begin() + d1, new_dims.begin() + d2);

        int n_ele = t->shape()[std::min(d1, d2)];

        const auto[col_strides, col_idxs] = generate_idxs(new_dims, std::max(d1, d2));     // based on new dims
        const auto[row_strides, row_idxs] = generate_idxs(t->shape(), std::min(d1, d2)); // based on old dims
        auto data = *t->data();
        // row_idxs.size() == col_idxs.size()
        for (int i = 0; i < row_idxs.size(); i++)
        {
            (*out_data)[std::slice(col_idxs[i], n_ele, col_strides[std::max(d1, d2)])] = data[std::slice(row_idxs[i], n_ele, row_strides[std::min(d1, d2)])];
        }

        auto output = std::make_shared<cyg::tensor<T>>(new_dims, out_data, t->requires_grad(), false);

        return output;
    }

    template<class T>
    std::shared_ptr<cyg::tensor<T>> var(const std::shared_ptr<cyg::tensor<T>>& base, int dim = INT_MAX, const int& correction=1, const bool& keepdim = true){
        auto data = *base->data();

        auto out_data = new std::valarray<T>();
        auto shape = base->shape();
        // @todo replace  with sum and broadcast ops 
        // (x- x_)**2 / n-1)
        if (dim == INT_MAX)
        {
            auto x_ = data.sum() / data.size();
            out_data->resize(1);
            (*out_data)[0] = std::pow(data - x_, 2).sum() / std::max(0, int(data.size() - correction));
            shape = {1};
        }
        else
        {
            dim<0? dim = shape.size() + dim : dim;
            const auto [strides, idxs] = generate_idxs(shape, dim);
            int n_elements = shape[dim];
            out_data->resize(idxs.size());
            for (int i = 0; i < idxs.size(); i++)
            {
                auto dim_data = std::valarray(data[std::slice(idxs[i], n_elements, strides[dim])]);
                auto m = std::pow((dim_data - dim_data.sum()/n_elements), 2).sum() / std::max(0, int(n_elements - correction));
                (*out_data)[i] = m;
            };
            shape[dim] = 1;
            if(!keepdim) shape.erase(shape.begin() + dim);
        }

        return std::make_shared<cyg::tensor<T>>(shape, out_data, base->requires_grad(), false);
    }

    template <class T>
    std::shared_ptr<cyg::tensor<T>> matmul(const std::shared_ptr<cyg::tensor<T>>& lhs, const std::shared_ptr<cyg::tensor<T>>& rhs)
    {

        const bool islhsgreater = lhs->rank() >= rhs->rank();
        std::vector<size_t> new_dims = islhsgreater ? lhs->shape() : rhs->shape();
        new_dims[islhsgreater ? new_dims.size() - 1 : new_dims.size() - 2] = islhsgreater ? rhs->shape()[rhs->rank() - 1] : lhs->shape()[lhs->rank() - 2];

        int n_elems = std::accumulate(new_dims.begin(), new_dims.end(), 1, std::multiplies<int>());

        auto out_data = new std::valarray<T>(n_elems);

        int lhs_dim = lhs->rank() - 1;
        int rhs_dim = rhs->rank() - 2;

        const auto [lhs_strides, lhs_rows] = generate_idxs(lhs->shape(), lhs_dim);
        const auto [rhs_strides, rhs_cols] = generate_idxs(rhs->shape(), rhs_dim);

        int n_cols = rhs->shape()[rhs_dim];

        auto ldata = *lhs->data();
        auto rdata = *rhs->data();

        //@todo improve matmul
        int ldx = 0;
        for(int rdx=0; rdx<n_elems; rdx++)
        {
            std::valarray<T> l_slice = std::valarray(ldata[std::slice(lhs_rows[ ldx % lhs_rows.size()], n_cols, lhs_strides[lhs_dim])]);
            std::valarray<T> r_slice = std::valarray(rdata[std::slice(rhs_cols[rdx % rhs_cols.size()], n_cols, rhs_strides[rhs_dim])]);
            (*out_data)[rdx] = (r_slice * l_slice).sum();
            ldx += ((rdx + 1) % rhs->shape()[rhs->rank() - 1]==0);
        }
        auto out_tensor = std::make_shared<cyg::tensor<T>>(new_dims, out_data, lhs->requires_grad() || rhs->requires_grad(), false);

        return out_tensor;
    }

    template <class T>
    std::shared_ptr<cyg::tensor<T>> mask(const std::shared_ptr<cyg::tensor<T>>& condition, const std::shared_ptr<cyg::tensor<T>>& true_value, const std::shared_ptr<cyg::tensor<T>>& false_value)
    {
        auto out_data = new std::valarray<T>(true_value->numel());
        auto req_grad = true_value->requires_grad() || false_value->requires_grad();
        auto new_dims = new std::vector<size_t>();

        if(is_broadcastable(condition->shape(), true_value->shape()) && is_broadcastable(condition->shape(), false_value->shape()))
        {
            auto cond_data = new std::valarray<T>(); auto true_data = new std::valarray<T>(); auto false_data = new std::valarray<T>();
            *cond_data = *condition->data(); *true_data = *true_value->data(); *false_data = *false_value->data();
            
            broadcast(cond_data, condition->shape(), true_data, true_value->shape(), new_dims);
            broadcast(cond_data, condition->shape(), false_data, false_value->shape(), new_dims);

            (*out_data)[*cond_data>0.0f] = (*true_data)[*cond_data>0.0f];
            (*out_data)[*cond_data<=0.0f] = (*false_data)[*cond_data<=0.0f];

            delete cond_data; delete true_data; delete false_data;
        }else 
        {
            auto cond = *condition->data();
            (*out_data)[cond>0.0f] = (*true_value->data())[cond>0.0f];
            (*out_data)[cond<=0.0f] = (*false_value->data())[cond<=0.0f];

        } 
        return std::make_shared<cyg::tensor<T>>(*new_dims, out_data, req_grad, false);
    }
}
#endif