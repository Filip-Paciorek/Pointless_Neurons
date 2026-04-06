#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "nn.h"

typedef struct {
	u32 num_in;
	u32 num_out;
	matrix* in;
	matrix* W;
	matrix* b;
	matrix* out;
} Layer;

typedef struct {
	u32 num_layers;
	Layer** layers;
} Network;

Network* create_network_structure(u32* array,u32 num_layers);
Layer* allocate_layer(u32 num_in, u32 num_out);
void calculate_layer(Layer* l1, matrix* input);
void free_layer(Layer* l);

int main()
{
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
	matrix* test = allocate_matrix(1,4);
	fill_matrix(test,0,4);
	Layer* l1 = allocate_layer(4,2);
	calculate_layer(l1,test);
	print_matrix(l1->out);
	free_matrix(test);
	free_layer(l1);
	//create_network_structure(l1);
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
	l->out = allocate_matrix(l->num_out,1);
	return l;
}

void calculate_layer(Layer* l,matrix* input)
{
	multiply_matrices(l->W,input,l->out);
}

void free_layer(Layer* l)
{
/* Free every part of the layer and then the layer itself*/
	free_matrix(l->in);
	free_matrix(l->W);
	free_matrix(l->b);
	free_matrix(l->out);
	free(l);
}

Network* create_network_structure(u32* arr, u32 num_layers)
{
	Network* nn = malloc(sizeof(*nn));
	nn->num_layers = num_layers-1;
	nn->layers = malloc(nn->num_layers*sizeof(Layer*));
	for(int i = 0; i <nn->num_layers;i++)
	{
		nn->layers[i] = allocate_layer(arr[i],arr[i+1]);
	}
	return nn;
}
void free_network(Network* nn)
{
	for(int i = 0; i < nn->num_layers;i++)
	{
		free_layer(nn->layers[i]);
	}
	free(nn->layers);
	free(nn);
}
i64 time_diff(struct timespec a, struct timespec b)
{
	return ((b.tv_sec - a.tv_sec) * 1000000000LL + (b.tv_nsec - a.tv_nsec));
}
