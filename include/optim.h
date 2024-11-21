#ifndef OPTIM_H
#define OPTIM_H
#include <unordered_map>
#include "tensor.h"

namespace optim
{

    typedef std::unordered_map<std::string, cyg::tptr<float>> pptr;

    class Optimizer
    {
        public:
            Optimizer(pptr* parameters, float lr): _parameters(parameters), _lr(lr){}
            virtual void step(){};
            void zero_grad(){
                for(const auto& [n, p]:*_parameters){
                    p->zero_grad();
                }
            };
     pptr* _parameters;
     float _lr;
    };



};
#endif