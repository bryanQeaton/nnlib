#include <chrono>
#include <iostream>
#include <vector>
#include "nnlib.h"
/*
Design paradigm:
1. the user ONLY ever passes single datapoints to the model, datasets are handled ENTIRELY by the user
2. external to the model, an explicit training function can be made which takes datasets, but that is seperate to the Model class.

todo:
save/load model
early stopping with patience
*/

int main() {
    // random dataset
    constexpr int dataset_scale=5;
    std::vector<float> rnd_data_x[1024*dataset_scale]={};
    std::vector<float> rnd_data_y[1024*dataset_scale]={};
    for (int i=0;i<1024*dataset_scale;i++) {
        for (int m=0;m<32;m++) {
            rnd_data_x[i].emplace_back(nn::random::rnd_normal(0,1));
        }
        float y=7.f;
        for (int m=0;m<16;m++) {
            y+=rnd_data_x[i][m];
        }
        for (int m=16;m<32;m++) {
            y-=rnd_data_x[i][m];
        }
        int f=y;
        f%=7;
        rnd_data_y[i].emplace_back(f);
    }
    auto model=nn::Model(
        32,
        {
            {256,nn::Relu,false},
            {64,nn::Relu,false},
            {1,nn::Linear,true}
        },
        nn::MSE,
        0.0001f,
        0.001f,
        true
        );
        nn::train(model,rnd_data_x,rnd_data_y,1024*dataset_scale,.1f,
            250,32, 2,true,
            .001,.0001,.6,.6
            );



    return 0;
}
