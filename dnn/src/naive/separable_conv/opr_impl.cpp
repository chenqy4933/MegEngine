/**
 * \file dnn/src/naive/separable_conv/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/separable_conv/opr_impl.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"
#include <cstring>

namespace megdnn {
namespace naive {
//using namespace sep_conv;

void SeparableConvForwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in filter_x,
        _megdnn_tensor_in filter_y,
        _megdnn_tensor_in dst,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, filter_x.layout, filter_y.layout, dst.layout, workspace.size);

    //Create kernel tensor
    int kw = filter_x.layout.shape[3];
    int kh = kw;
    int ic = filter_x.layout.shape[1];
    int oc = filter_x.layout.shape[0];

    TensorLayout kerLayout({(size_t)oc, (size_t)ic, (size_t)kh, (size_t)kw}, dtype::Float32());
    void* filter2d_buf = malloc(oc * ic * kh * kw * sizeof(float));
    TensorND filter2d(filter2d_buf, kerLayout);
    float* kerx = (float*)filter_x.raw_ptr;
    float* kery = (float*)filter_y.raw_ptr;
    float* ker2d = (float*)filter2d_buf;
    
    // Generate 2D-filter    
    int k_pos = 0;
    for(int cn = 0; cn < ic * oc ; ++cn) {
    	for(int h = 0; h < kh; ++h) {
    		for (int w = 0; w < kw; ++w) {
    			ker2d[ k_pos ++] = kerx[w] * kery[h];
    		}
    	}
    	kerx += kw;
    	kery += kw;
    }
    
    ConvolutionForwardImpl* convOptr  = new ConvolutionForwardImpl(this->handle());
    Workspace empty_wsp;
    convOptr->exec(src, filter2d, dst, empty_wsp);
    delete(convOptr);

    free(filter2d_buf);
}

} // namespace naive
} // namespace megdnn

// vim: syntax=cpp.doxygen
