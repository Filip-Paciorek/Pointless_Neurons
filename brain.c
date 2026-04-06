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
	Layer* layers;
} Network;

void create_network_structure(Layer* array);
Layer* allocate_layer(u32 num_in, u32 num_out);
Layer* calculate_layer(Layer* l1, Layer* l2);
void free_layer(Layer* l);

int main()
{
	matrix* mat = allocate_matrix(2,4);
	fill_matrix(mat,0,3);
	//print_matrix(mat);
	printf("\n-------------\n"); 
	matrix* mat_t = transpose_matrix(mat);
	//print_matrix(mat_t);
	matrix* mult_mat = multiply_matrices(mat,mat_t);
	printf("\n----------------\n");
	//print_matrix(mult_mat);
	free_matrix(mat);
	free_matrix(mat_t);
	free_matrix(mult_mat);
	Layer* l1 = allocate_layer(32,16);
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
	l->W = allocate_matrix(l->num_in,l->num_out);
	l->b = allocate_matrix(l->num_out,1);
	l->out = allocate_matrix(l->num_out,1);
	return l;
}

Layer* calculate_layer(Layer* l1, Layer* l2)
{
	multiply_matrices(l1->out,l2->W);
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

void create_network_structure(Layer* array)
{
	
	printf("");
}

i64 time_diff(struct timespec a, struct timespec b)
{
	return ((b.tv_sec - a.tv_sec) * 1000000000LL + (b.tv_nsec - a.tv_nsec));
}
