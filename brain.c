#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "nn.h"

//struct definitions

typedef struct {
	u32 num_in;
	u32 num_out;
	matrix* in;
	matrix* W;
	matrix* dW;
	matrix* b;
	matrix* db;
	matrix* z;
	matrix* out;
} Layer;

typedef struct {
	u32 num_layers;
	Layer** layers;
} Network;

//function definitions

Layer* allocate_layer(u32 num_in, u32 num_out, u32 batch_size);
void print_layer_weights(Layer* l);
void print_layer_outputs(Layer* l);
void calculate_layer(Layer* l1, matrix* input);
void free_layer(Layer* l);
Network* create_network_structure(u32* array,u32 num_layers,u32 batch_size);
void print_network_params(Network* nn);
void free_network(Network* nn);
void b_initialization(Layer* l);
void He_initialization(Layer* l);
void Xavier_initialization(Layer* l);
void ReLU_activation(Layer* l);
void Sigmoid_activation(Layer* l);
void initialize_network(Network* nn);
//BACKPROP FUNCTIONS
float MSE(matrix* Y, Layer* Y_PRED);
matrix* MSE_derivative_W(matrix* Y,Layer* Y_PRED);
matrix* MSE_derivative_b(matrix* Y,Layer* Y_PRED);
void Gradient_Descent(Layer* l,matrix* Y,u32 learning_rate);
int main()
{
	//testing basic matrix arithmetic
	matrix* mat = allocate_matrix(2,4);
	fill_matrix(mat,0,3);
	matrix* mat_t = allocate_matrix(4,2);
	transpose_matrix(mat,mat_t);
	matrix* mult_mat = allocate_matrix(2,2);
	multiply_matrices(mat,mat_t,mult_mat);
	matrix* add_mat = allocate_matrix(2,4);
	matrix* add_vec = allocate_matrix(1,4);
	fill_matrix(add_mat,0,5);
	fill_matrix(add_vec,0,5);
	add_matrices(mat,add_mat,add_mat);
	add_vector_to_matrix(add_mat,add_vec,add_mat);
	free_matrix(mat);
	free_matrix(mat_t);
	free_matrix(mult_mat);
	free_matrix(add_mat);
	free_matrix(add_vec);
	//testing nn structures
	matrix* test = allocate_matrix(1,4);
	fill_matrix(test,0,4);
	Layer* l1 = allocate_layer(4,2,1);
	calculate_layer(l1,test);
	print_matrix(l1->out);
	free_matrix(test);
	free_layer(l1);
	u32 arr[] = {12,8,2};
	Network* nn = create_network_structure(arr,3,1);
	//fill with gibberish
	fill_matrix(nn->layers[0]->in,0,10);
	initialize_network(nn);
	print_network_params(nn);
	//for each layer
	for(int i=0; i < nn->num_layers;i++)
	{
		printf("Layer %d \n", i);
		//calculate z 
		calculate_layer(nn->layers[i],nn->layers[i]->in);
		//activate z giving us the output neurons
		if (i < nn->num_layers -1)
		{

			ReLU_activation(nn->layers[i]);
		//make sure there is nothing in the layer we want to output our data to
			free_matrix(nn->layers[i+1]->in);
		//place our output in the input of the next layer
			nn->layers[i+1]->in = nn->layers[i]->out;
		}
		else
		{
			Sigmoid_activation(nn->layers[i]);
     		}
		//print_layer_weights(nn->layers[i]);
		//print_layer_outputs(nn->layers[i]);
		print_matrix(nn->layers[i]->out);
	};
	free_network(nn);
	return 1;
}

Layer* allocate_layer(u32 num_in,u32 num_out, u32 batch_size)
{
/* Using arguments num_in and num_out calculate the sizes of each element of the layer necessary */
	Layer* l = malloc(sizeof(*l));
	//define the sizes of the input and the output of the layer
	l->num_in = num_in;
	l->num_out = num_out;
	//the input layer takes in the input of size num_in
	l->in = allocate_matrix(l->num_in,batch_size);
	//then it gets multiplied by the number of neurons it has connections to, so that for each
	//we get a separate piece of space to fit datapoints in
	l->W = allocate_matrix(l->num_out,l->num_in);
	//do the same for the derivative as it has the same size
	l->dW = allocate_matrix(l->num_out,l->num_in);
	//for b we just need the space for number of output neurons as it is just addition to the final sum and we will use broadcasting for each datapoint
	l->b = allocate_matrix(l->num_out,1);
	//do the same for the derivative as it has the same size
	l->db = allocate_matrix(l->num_out,1);
	//for z we need the size of output * datapoints processed
	l->z = allocate_matrix(l->num_out,batch_size);
	//we allocate space for the output neurons
	l->out = allocate_matrix(l->num_out,batch_size);
	return l;
}
void print_layer_weights(Layer* l)
{
/* Print out the weigths assigned to the layer*/
	for(int i =0; i < l->W->rows;i++)
	{
		for(int j = 0; j < l->W->columns;j++)
		{
			printf("%f ",l->W->data[i*l->W->columns + j]);
		}
		printf("\n");
	}
}
void print_layer_outputs(Layer* l)
{
	for(int i = 0; i < l->num_out; i++)
	{
		printf("%f",l->z->data[i]);
	}
	printf("\n");
}
void add_bias_to_z(matrix* z, matrix* b)
{
	for(u32 i = 0; i < z->columns; i++)
	{
		for(u32 j = 0; j < z->rows; j++)
		{
			z->data[j * z->columns + i] += b->data[j];
		}

	}
}
void calculate_layer(Layer* l,matrix* input)
{
/* Calculate the layer by multiplication and point it to the output*/
	//In l->W we hold the input weights for ith output neuron in ith row (its flattened but the rules apply still)
	//after multiplication we are left with the sum of "firing" for each neuron in each row
	//where our matrix is of size output*batch_size
	multiply_matrices(l->W,input,l->z);
	//then we add our bias to the already calculated "firings" but for ALL data
	add_bias_to_z(l->z,l->b);
}

void free_layer(Layer* l)
{
/* Free every part of the layer and then the layer itself*/
	if (l->in)
	{
		free_matrix(l->in);
	}
	free_matrix(l->W);
	free_matrix(l->dW);
	free_matrix(l->b);
	free_matrix(l->db);
	free_matrix(l->z);
	free_matrix(l->out);
	free(l);
}

Network* create_network_structure(u32* arr, u32 num_layers, u32 batch_size)
{
/* Allocate a network structure with a 2D layer structure*/
	Network* nn = malloc(sizeof(*nn));
	//set number of layers to num_layers -1 because last "layer" is just output of previous layer 
	nn->num_layers = num_layers-1;
	//set up memory for the layers to be held in
	nn->layers = malloc(nn->num_layers*sizeof(Layer*));
	for(int i = 0; i <nn->num_layers;i++)
	{
		//set up memory for each layer
		nn->layers[i] = allocate_layer(arr[i],arr[i+1],batch_size);
	}
	return nn;
}
void print_network_params(Network* nn)
{
/* Print network's layer's number of neurons going in and out*/
	for(int i = 0; i < nn->num_layers;i++)
	{
		printf("Layer %d - in: %d out: %d \n",i,nn->layers[i]->num_in,nn->layers[i]->num_out);
	}
}
void free_network(Network* nn)
{
/* Free the network */
	for(int i = 0; i < nn->num_layers;i++)
	{
		if (i >0)
		{
			nn->layers[i]->in = NULL;
		}
		free_layer(nn->layers[i]);
	}
	free(nn->layers);
	free(nn);
}
void b_initialization(Layer* l)
{
	fill_matrix(l->b,0,1);
}
void He_initialization(Layer* l)
{
/* Use He initialization on all the weights of the layer*/
	double limit = sqrtf((2.0 / l->num_in));
	for (int i =0; i < l->W->rows;i++)
	{
		for(int j =0; j < l->W->columns;j++)
		{
			l->W->data[i*l->W->columns + j] = (((double)rand() /(double)RAND_MAX) * (2*limit))-limit;
		}
	}
}
void Xavier_initialization(Layer* l)
{
	u32 connections = l->num_in + l->num_out;
	float limit = sqrtf(6.0/connections);
	for (int i =0; i < l->W->rows;i++)
	{
		for(int j =0; j < l->W->columns;j++)
		{
			l->W->data[i*l->W->columns + j] = (((double)rand() /(double)RAND_MAX) * (2*limit))-limit;
		}
	}
}
void ReLU_activation(Layer* l)
{
	u32 all = l->z->rows * l->z->columns;
	for(int i = 0;i < all;i++)
	{
		if (l->z->data[i] > 0)
		{
			l->out->data[i] = l->z->data[i];
		}
     		else
     		{
			l->out->data[i] = 0;
		}
     
	}
}
void Sigmoid_activation(Layer* l)
{
	u32 all = l->z->rows * l->z->columns;
	for(int i = 0;i < all;i++)
	{
		l->out->data[i] = 1/(1+exp(-l->z->data[i]));
	}
}
void initialize_network(Network* nn)
{
	for (int i = 0; i <nn->num_layers;i++)
	{
		if (i < nn->num_layers -1) 
		{
			He_initialization(nn->layers[i]);
			b_initialization(nn->layers[i]);
		}
      		else
		{
			Xavier_initialization(nn->layers[i]);
			b_initialization(nn->layers[i]);
		}
	}

}
float MSE(matrix* Y, Layer* Y_PRED)
{
	if (Y->rows != Y_PRED->out->rows)
	{
		return -1;
	}
	double sum = 0.0;
	u32 n = (Y_PRED->out->rows*Y_PRED->out->columns);
	for (u32 i = 0; i < n; i++)
	{
		double diff = (Y_PRED->out->data[i] - Y->data[i]);
		sum += diff * diff;	
      }
	return sum / (double)n;
}
matrix* MSE_derivative_W(matrix* Y, Layer* Y_PRED)
{
	return 0;
}
matrix* MSE_derivative_b(matrix* Y, Layer* Y_PRED)
{

	for(u32 i = 0; i < Y->rows; i++)
	{
		for (u32 j = 0; j < Y->columns; j++)
		{
			i32 d = (Y_PRED->z->data[i*Y->columns + j] - Y->data[i*Y->columns + j]);
		}
	}
}
i64 time_diff(struct timespec a, struct timespec b)
{
	return ((b.tv_sec - a.tv_sec) * 1000000000LL + (b.tv_nsec - a.tv_nsec));
}
