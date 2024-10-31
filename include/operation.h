#ifndef OPERATION_H
#define OPERATION_H
#include <map>
#include <valarray>
#include <vector>
#include <memory>

namespace cyg
{
    // template <typename T>
    //  void op_cpu(std::valarray<float>& out, const std::valarray<float>& lhs, const std::valarray<float>& rhs, T operation);

    template <class T>
    class Context
    {
        std::vector<std::shared_ptr<T>> cache; // check T destructor, vector deallocates,
    public:
        Context() {}
        std::map<std::string, int> saved_data;

        /**
         * @brief Save input tensors[shared pointers] that can be retrieved later.
         * the Ts can be retrieved by calling the get_saved_variables() method
         * for ex.
         *  const vector<int>dims{1,2,3};
         *  auto arg1 = ones(dims);
         *  auto arg2 = ones(dims);
         *  ctx->save_for_backward({arg1, arg2});
         *
         * @param Ts(type std::vector<std::shared_ptr<T>>)
         */
        void save_for_backward(std::vector<std::shared_ptr<T>> Ts)
        {
            for (const auto &t : Ts)
                this->cache.push_back(t);
        };
        /**
         * @brief Retrieve tensors that have been saved previously.
         * for ex.
         *  const vector<int>dims{1,2,3};
         *  auto arg1 = ones(dims);
         *  auto arg2 = ones(dims);
         *  ctx->save_for_backward({arg1, arg2});
         *  ctx->get_variables()=={arg1, arg2};
         *
         * @return the vectors of Ts shared_pointers (type std::vector<std::shared_ptr<tensor>>)
         */
        std::vector<std::shared_ptr<T>> get_variables() { return this->cache; };
        // Context(const Context<T> &context) : cache(context->cache), saved_data(context->saved_data) {} // rule of three/five/zero
        ~Context() noexcept
        {
            this->saved_data.clear();
        }
    };

    template <class T>
    class Operation : public std::enable_shared_from_this<Operation<T>>
    {
    protected:
        std::unique_ptr<Context<T>> context;

    public:
        Operation()
        {
            this->context = std::make_unique<Context<T>>();
        };
        virtual void backward(std::valarray<float> *incoming_grad) {};
        // Operation(const Operation &operation) : context(std::move(operation->context)) {} // rule of three/five/zero
        ~Operation() noexcept
        {
            this->context.reset();
        }
    };

    template <class T>
    class Add : public Operation<T>
    {
    public:
        Add() : Operation<T>() {}
        std::shared_ptr<T> forward(const std::shared_ptr<T> &lhs, const std::shared_ptr<T> &rhs);
        void backward(std::valarray<float> *incoming_grad) override;
    };

    template <class T>
    class Mul : public Operation<T>
    {
    public:
        Mul() : Operation<T>() {}
        std::shared_ptr<T> forward(const std::shared_ptr<T> &lhs, const std::shared_ptr<T> &rhs);
        void backward(std::valarray<float> *incoming_grad) override;
    };
    template <class T>
    class Div : public Operation<T>
    {
    public:
        Div() : Operation<T>() {}
        std::shared_ptr<T> forward(const std::shared_ptr<T> &numerator, const std::shared_ptr<T> &denominator);
        void backward(std::valarray<float> *incoming_grad) override;
    };

    template <class T>
    class Pow : public Operation<T>
    {
    public:
        Pow() : Operation<T>() {}
        std::shared_ptr<T> forward(const std::shared_ptr<T> &base, const std::shared_ptr<T> &exponent);
        void backward(std::valarray<float> *incoming_grad) override;
    };
    template <class T>
    class Mean : public Operation<T>
    {
    public:
        Mean() : Operation<T>() {}
        std::shared_ptr<T> forward(const std::shared_ptr<T> &t, const int &dim, const bool &keepdim = true);
        void backward(std::valarray<float> *incoming_grad) override;
    };
    template <class T>
    class Exp : public Operation<T>
    {
    public:
        Exp() : Operation<T>() {}
        std::shared_ptr<T> forward(const std::shared_ptr<T> &exponent);
        void backward(std::valarray<float> *incoming_grad) override;
    };
    template <class T>
    class Log : public Operation<T>
    {
    public:
        Log() : Operation<T>() {}
        std::shared_ptr<T> forward(const std::shared_ptr<T> &v1);
        void backward(std::valarray<float> *incoming_grad) override;
    };

    template <class T>
    class Sum : public Operation<T>
    {
    public:
        Sum() : Operation<T>() {}
        std::shared_ptr<T> forward(const std::shared_ptr<T> &v1, const int &dim = -1, const bool &keepdim = true);
        void backward(std::valarray<float> *incoming_grad) override;
    };

    // template <class T>
    // class Cos : public Operation<T>
    // {
    // public:
    //     Cos() : Operation<T>() {}
    //     std::shared_ptr<T> forward(const std::shared_ptr<T>& v1);
    //     void backward(std::valarray<float> *incoming_grad) override;
    // };

    // template <class T>
    // class Sin : public Operation<T>
    // {
    // public:
    //     Sin() : Operation<T>() {}
    //     std::shared_ptr<T> forward(const std::shared_ptr<T>& v1);
    //     void backward(std::valarray<float> *incoming_grad) override;
    // };
}
#endif