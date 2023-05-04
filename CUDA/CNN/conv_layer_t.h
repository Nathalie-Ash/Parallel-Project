
#pragma once
#include "layer_t.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#pragma pack(push, 1)

__global__ void convolve_kernel(float* in, float* filters, float* out, int width, int height, int depth, int filter_size, int num_filters)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int filter = blockIdx.z;

    if (x < width && y < height)
    {
        float sum = 0;
        for (int i = 0; i < filter_size; i++)
        {
            for (int j = 0; j < filter_size; j++)
            {
                for (int z = 0; z < depth; z++)
                {
                    float f = filters[(filter * filter_size * filter_size * depth) + (i * filter_size * depth) + (j * depth) + z];
                    float v = in[((y + j) * (width + filter_size - 1) * depth) + ((x + i) * depth) + z];
                    sum += f * v;
                }
            }
        }
        out[(filter * width * height) + (y * width) + x] = sum;
    }
}


struct conv_layer_t
{
	layer_type type = layer_type::conv;
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;
	std::vector<tensor_t<float>> filters;
	std::vector<tensor_t<gradient_t>> filter_grads;
	uint16_t stride;
	uint16_t extend_filter;

	conv_layer_t( uint16_t stride, uint16_t extend_filter, uint16_t number_filters, tdsize in_size )
		:
		grads_in( in_size.x, in_size.y, in_size.z ),
		in( in_size.x, in_size.y, in_size.z ),
		out(
		(in_size.x - extend_filter) / stride + 1,
			(in_size.y - extend_filter) / stride + 1,
			number_filters
		)

	{
		this->stride = stride;
		this->extend_filter = extend_filter;
		assert( (float( in_size.x - extend_filter ) / stride + 1)
				==
				((in_size.x - extend_filter) / stride + 1) );

		assert( (float( in_size.y - extend_filter ) / stride + 1)
				==
				((in_size.y - extend_filter) / stride + 1) );

		for ( int a = 0; a < number_filters; a++ )
		{
			tensor_t<float> t( extend_filter, extend_filter, in_size.z );

			int maxval = extend_filter * extend_filter * in_size.z;

			for ( int i = 0; i < extend_filter; i++ )
				for ( int j = 0; j < extend_filter; j++ )
					for ( int z = 0; z < in_size.z; z++ )
						t( i, j, z ) = 1.0f / maxval * rand() / float( RAND_MAX );
			filters.push_back( t );
		}
		for ( int i = 0; i < number_filters; i++ )
		{
			tensor_t<gradient_t> t( extend_filter, extend_filter, in_size.z );
			filter_grads.push_back( t );
		}

	}

	point_t map_to_input( point_t out, int z )
	{
		out.x *= this->stride;
		out.y *= this->stride;
		out.z = z;
		return out;
	}

	struct range_t
	{
		int min_x, min_y, min_z;
		int max_x, max_y, max_z;
	};

	int normalize_range( float f, int max, bool lim_min )
	{
		if ( f <= 0 )
			return 0;
		max -= 1;
		if ( f >= max )
			return max;

		if ( lim_min ) // left side of inequality
			return ceil( f );
		else
			return floor( f );
	}

	range_t map_to_output( int x, int y )
	{
		float a = x;
		float b = y;
		return
		{
			normalize_range( (a - extend_filter + 1) / stride, out.size.x, true ),
			normalize_range( (b - extend_filter + 1) / stride, out.size.y, true ),
			0,
			normalize_range( a / stride, out.size.x, false ),
			normalize_range( b / stride, out.size.y, false ),
			(int)filters.size() - 1,
		};
	}

	void activate( tensor_t<float>& in )
	{
		this->in = in;
		activate();
	}


void activate()
{
    float* d_in, * d_filters, * d_out;

    int in_size = (in.size.x + extend_filter - 1) * (in.size.y + extend_filter - 1) * in.size.z;
    int filter_size = extend_filter * extend_filter * in.size.z;
    int out_size = out.size.x * out.size.y * filters.size();

    cudaMalloc(&d_in, in_size * sizeof(float));
    cudaMemcpy(d_in, &in(0, 0, 0), in_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_filters, filter_size * filters.size() * sizeof(float));
    for (int i = 0; i < filters.size(); i++)
    {
        cudaMemcpy(&d_filters[i * filter_size], &filters[i](0, 0, 0), filter_size * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&d_out, out_size * sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((out.size.x + threadsPerBlock.x - 1) / threadsPerBlock.x, (out.size.y + threadsPerBlock.y - 1) / threadsPerBlock.y, filters.size());

    convolve_kernel<<<numBlocks, threadsPerBlock>>>(d_in, d_filters, d_out, in.size.x, in.size.y, in.size.z, extend_filter, filters.size());

    cudaMemcpy(&out(0, 0, 0), d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_filters);
    cudaFree(d_out);
}


	void fix_weights()
	{
		for ( int a = 0; a < filters.size(); a++ )
			for ( int i = 0; i < extend_filter; i++ )
				for ( int j = 0; j < extend_filter; j++ )
					for ( int z = 0; z < in.size.z; z++ )
					{
						float& w = filters[a].get( i, j, z );
						gradient_t& grad = filter_grads[a].get( i, j, z );
						w = update_weight( w, grad );
						update_gradient( grad );
					}
	}

	void calc_grads( tensor_t<float>& grad_next_layer )
	{

		for ( int k = 0; k < filter_grads.size(); k++ )
		{
			for ( int i = 0; i < extend_filter; i++ )
				for ( int j = 0; j < extend_filter; j++ )
					for ( int z = 0; z < in.size.z; z++ )
						filter_grads[k].get( i, j, z ).grad = 0;
		}

		for ( int x = 0; x < in.size.x; x++ )
		{
			for ( int y = 0; y < in.size.y; y++ )
			{
				range_t rn = map_to_output( x, y );
				for ( int z = 0; z < in.size.z; z++ )
				{
					float sum_error = 0;
					for ( int i = rn.min_x; i <= rn.max_x; i++ )
					{
						int minx = i * stride;
						for ( int j = rn.min_y; j <= rn.max_y; j++ )
						{
							int miny = j * stride;
							for ( int k = rn.min_z; k <= rn.max_z; k++ )
							{
								int w_applied = filters[k].get( x - minx, y - miny, z );
								sum_error += w_applied * grad_next_layer( i, j, k );
								filter_grads[k].get( x - minx, y - miny, z ).grad += in( x, y, z ) * grad_next_layer( i, j, k );
							}
						}
					}
					grads_in( x, y, z ) = sum_error;
				}
			}
		}
	}
};
#pragma pack(pop)