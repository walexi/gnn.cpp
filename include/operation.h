#ifndef OPERATION_H
#define OPERATION_H
#include <map>
#include <vector>
#include <memory>
namespace cyg
{
    template<class T>
    class Context
    {
        std::vector<std::shared_ptr<T>> cache; // check T destructor, vector deallocates, 
    public:
        Context(){}
        std::map<std::string, int> saved_data;
        void save_for_backward(std::vector<std::shared_ptr<T>> Ts);
        std::vector<std::shared_ptr<T>> get_variables();
        Context(const Context& context):cache(context.cache), saved_data(context.saved_data){} //rule of three/five/zero
        ~Context()noexcept{
            this->saved_data.clear();
        }
    };

    template<class T>
    class Operation : public std::enable_shared_from_this<Operation<T>>
    {
        protected:
           std::unique_ptr<Context<T>> context;
        public:
            Operation(){
                this->context = std::make_unique<Context<T>>();
            };
            //not passing shared_ptr by const ref cos I need to keep the Ts alive so i can update their grad during the backward pass
            //basically operation should share ownership of input Ts since Ts's grad will be updated in the backward passes
            //returning shared_ptr instead of raw or weak ptr, since the lifetime of the returned was initiated within the function block
            //its safer to increment the ref count as the func returns to the caller, weak_ptr can also work here too though
            virtual std::shared_ptr<T> forward(const std::shared_ptr<T>& lhs, const std::shared_ptr<T>& rhs = nullptr);
            virtual void backward(std::vector<float>* incoming_grad);
            Operation(const Operation& operation):context(operation->context){}
            ~Operation() noexcept{
                this->context.reset();
            }
    };

    template<class T>
    class Add : public Operation<T>{
        public:
            Add(): Operation<T>(){}
            std::shared_ptr<T> forward(const std::shared_ptr<T>& lhs, const std::shared_ptr<T>& rhs) override;
            void backward(std::vector<float>* incoming_grad) override;
    };
    
    template<class T>
    class Mul : public Operation<T>{
        public:
            Mul(): Operation<T>(){}
            std::shared_ptr<T> forward(const std::shared_ptr<T>& lhs, const std::shared_ptr<T>& rhs) override;
            void backward(std::vector<float>* incoming_grad) override;
    };
    template<class T>
    class Div : public Operation<T>{
        public:
            Div(): Operation<T>(){}
            std::shared_ptr<T> forward(const std::shared_ptr<T>& numerator, const std::shared_ptr<T>& denominator) override;
            void backward(std::vector<float>* incoming_grad) override;
    };
}
#endif