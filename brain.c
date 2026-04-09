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
	matrix* b;
	matrix* z;
	matrix* out;
} Layer;

typedef struct {
	u32 num_layers;
	Layer** layers;
} Network;

//function definitions

Layer* allocate_layer(u32 num_in, u32 num_out);
void print_layer_weights(Layer* l);
void print_layer_outputs(Layer* l);
void calculate_layer(Layer* l1, matrix* input);
void free_layer(Layer* l);
Network* create_network_structure(u32* array,u32 num_layers);
void print_network_params(Network* nn);
void free_network(Network* nn);
void b_initialization(Layer* l);
void He_initialization(Layer* l);
void Xavier_initialization(Layer* l);
void ReLU_activation(Layer* l);
void Sigmoid_activation(Layer* l);
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
	Layer* l1 = allocate_layer(4,2);
	calculate_layer(l1,test);
	print_matrix(l1->out);
	free_matrix(test);
	free_layer(l1);
	u32 arr[] = {12,8,2};
	Network* nn = create_network_structure(arr,3);
	fill_matrix(nn->layers[0]->in,0,10);
	print_network_params(nn);
	for(int i=0; i < nn->num_layers-1;i++)
	{
		printf("Layer %d \n", i);
		b_initialization(nn->layers[i]);
		He_initialization(nn->layers[i]);
		calculate_layer(nn->layers[i],nn->layers[i]->in);
		ReLU_activation(nn->layers[i]);
		//print_layer_weights(nn->layers[i]);
		//print_layer_outputs(nn->layers[i]);
		print_matrix(nn->layers[i]->out);
	};
	b_initialization(nn->layers[nn->num_layers-1]);
	Xavier_initialization(nn->layers[nn->num_layers-1]);
	calculate_layer(nn->layers[nn->num_layers-1],nn->layers[nn->num_layers-1]->in);
	Sigmoid_activation(nn->layers[nn->num_layers-1]);
	free_network(nn);
	return 1;
}

Layer* allocate_layer(u32 num_in,u32 num_out)
{
/* Using arguments num_in and num_out calculate the sizes of each element of the layer necessary */
	Layer* l = malloc(sizeof(*l));
	l->num_in = num_in;
	l->num_out = num_out;
	l->in = allocate_matrix(l->num_in,1);
	l->W = allocate_matrix(l->num_out,l->num_in);
	l->b = allocate_matrix(l->num_out,1);
	l->z = allocate_matrix(l->num_out,l->num_in);
	l->out = allocate_matrix(l->num_out,1);
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
		printf("%f",l->z[i]);
	}
	printf("\n");
}
void calculate_layer(Layer* l,matrix* input)
{
/* Calculate the layer by multiplication and point it to the output*/
	multiply_matrices(l->W,input,l->z);
}

void free_layer(Layer* l)
{
/* Free every part of the layer and then the layer itself*/
	free_matrix(l->in);
	free_matrix(l->W);
	free_matrix(l->b);
	free_matrix(l->z);
	free_matrix(l->out);
	free(l);
}

Network* create_network_structure(u32* arr, u32 num_layers)
{
/* Allocate a network structure with a 2D layer structure*/
	Network* nn = malloc(sizeof(*nn));
	nn->num_layers = num_layers-1;
	nn->layers = malloc(nn->num_layers*sizeof(Layer*));
	for(int i = 0; i <nn->num_layers;i++)
	{
		nn->layers[i] = allocate_layer(arr[i],arr[i+1]);
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
	for(int i = 0;i < l->num_out;i++)
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
	for(int i = 0;i < l->num_out;i++)
	{
		l->out->data[i] = 1/(1+exp(-l->z->data[i]));
	}
}
i64 time_diff(struct timespec a, struct timespec b)
{
	return ((b.tv_sec - a.tv_sec) * 1000000000LL + (b.tv_nsec - a.tv_nsec));
}
