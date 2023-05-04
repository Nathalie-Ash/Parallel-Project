#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "layer_t.h"

#pragma pack(push, 1)

__global__
void activate_kernel(float* in, float* weights, float* out, int in_x, int in_y, int in_z, int out_x)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < out_x)
    {
        float inputv = 0;

        for (int i = 0; i < in_x; i++)
        {
            for (int j = 0; j < in_y; j++)
            {
                for (int z = 0; z < in_z; z++)
                {
                    int m = z * (in_x * in_y) + j * (in_x) + i;
                    inputv += in[m] * weights[m * out_x + n];
                }
            }
        }

        out[n] = 1.0f / (1.0f + expf(-inputv));
    }
}

struct fc_layer_t
{
	layer_type type = layer_type::fc;
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;
	std::vector<float> input;
	tensor_t<float> weights;
	std::vector<gradient_t> gradients;

	fc_layer_t( tdsize in_size, int out_size )
		:
		in( in_size.x, in_size.y, in_size.z ),
		out( out_size, 1, 1 ),
		grads_in( in_size.x, in_size.y, in_size.z ),
		weights( in_size.x*in_size.y*in_size.z, out_size, 1 )
	{
		input = std::vector<float>( out_size );
		gradients = std::vector<gradient_t>( out_size );


		int maxval = in_size.x * in_size.y * in_size.z;

		for ( int i = 0; i < out_size; i++ )
			for ( int h = 0; h < in_size.x*in_size.y*in_size.z; h++ )
				weights( h, i, 0 ) = 2.19722f / maxval * rand() / float( RAND_MAX );
		// 2.19722f = f^-1(0.9) => x where [1 / (1 + exp(-x) ) = 0.9]
	}

	float activator_function( float x )
	{
		//return tanhf( x );
		float sig = 1.0f / (1.0f + exp( -x ));
		return sig;
	}

	float activator_derivative( float x )
	{
		//float t = tanhf( x );
		//return 1 - t * t;
		float sig = 1.0f / (1.0f + exp( -x ));
		return sig * (1 - sig);
	}

	void activate( tensor_t<float>& in )
	{
		this->in = in;
		activate();
	}

	int map( point_t d )
	{
		return d.z * (in.size.x * in.size.y) +
			d.y * (in.size.x) +
			d.x;
	}

void activate()
{
    float* in_dev;
    float* weights_dev;
    float* out_dev;
    cudaMalloc(&in_dev, in.size.x * in.size.y * in.size.z * sizeof(float));
    cudaMalloc(&weights_dev, in.size.x * in.size.y * in.size.z * out.size.x * sizeof(float));
    cudaMalloc(&out_dev, out.size.x * sizeof(float));

    cudaMemcpy(in_dev, in.data, in.size.x * in.size.y * in.size.z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weights_dev, weights.data, in.size.x * in.size.y * in.size.z * out.size.x * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int num_blocks = (out.size.x + threads_per_block - 1) / threads_per_block;

    activate_kernel<<<num_blocks, threads_per_block>>>(in_dev, weights_dev, out_dev, in.size.x, in.size.y, in.size.z, out.size.x);
    cudaDeviceSynchronize();

    cudaMemcpy(out.data, out_dev, out.size.x * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(in_dev);
    cudaFree(weights_dev);
    cudaFree(out_dev);
}


	void fix_weights()
	{
		for ( int n = 0; n < out.size.x; n++ )
		{
			gradient_t& grad = gradients[n];
			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int z = 0; z < in.size.z; z++ )
					{
						int m = map( { i, j, z } );
						float& w = weights( m, n, 0 );
						w = update_weight( w, grad, in( i, j, z ) );
					}

			update_gradient( grad );
		}
	}

	void calc_grads( tensor_t<float>& grad_next_layer )
	{
		memset( grads_in.data, 0, grads_in.size.x *grads_in.size.y*grads_in.size.z * sizeof( float ) );
		for ( int n = 0; n < out.size.x; n++ )
		{
			gradient_t& grad = gradients[n];
			grad.grad = grad_next_layer( n, 0, 0 ) * activator_derivative( input[n] );

			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int z = 0; z < in.size.z; z++ )
					{
						int m = map( { i, j, z } );
						grads_in( i, j, z ) += grad.grad * weights( m, n, 0 );
					}
		}
	}
};
#pragma pack(pop)
