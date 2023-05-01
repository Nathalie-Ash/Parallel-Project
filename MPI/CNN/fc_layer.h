#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "layer_t.h"

#include <mpi.h>

#pragma pack(push, 1)
//double elapsed_time;
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
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_rows = in.size.x / size;
    int start_row = rank * local_rows;
    int end_row = start_row + local_rows - 1;

    tensor_t<float> local_in(local_rows, in.size.y, in.size.z);
    tensor_t<float> local_out(out.size.x / size, 1, 1);

    // Scatter the input tensor among the MPI processes
    MPI_Scatter(in.data, local_rows * in.size.y * in.size.z, MPI_FLOAT,
                local_in.data, local_rows * in.size.y * in.size.z, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // Compute the matrix multiplication locally
    for (int n = 0; n < out.size.x; n += size)
    {
        for (int i = start_row; i <= end_row; i++)
        {
            for (int j = 0; j < in.size.y; j++)
            {
                for (int z = 0; z < in.size.z; z++)
                {
                    int m = map({i, j, z});
                    int col = n + rank;
                    local_out(col / size, 0, 0) += local_in(i - start_row, j, z) * weights(m, col, 0);
                }
            }
        }
    }

    // Gather the output tensor to the root process
    MPI_Gather(local_out.data, local_out.size.x, MPI_FLOAT,
               out.data, local_out.size.x, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    // Apply the activation function to the output tensor
    if (rank == 0)
    {
        for (int n = 0; n < out.size.x; n++)
        {
            out(n, 0, 0) = activator_function(out(n, 0, 0));
        }
    }    
    
    //printf("Fully Connected: %f\n", elapsed_time);
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
