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
	matrix* dz;
	matrix* out;
} Layer;

typedef struct {
	u32 num_layers;
	Layer** layers;
} Network;

//function definitions
void substract_matrices(matrix* mat1, matrix* mat2, matrix* sub_mat);
void fill_matrix_with_value(matrix* mat1, double value);
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
void MSE_derivative_W(matrix* Y,Layer* Y_PRED);
void MSE_derivative_b(matrix* Y,Layer* Y_PRED);
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
	matrix* Y = allocate_matrix(1, 2); 
	Y->data[0] = 1.0; Y->data[1] = 0.0;

	Layer* l_test = allocate_layer(4, 1, 2); 
	l_test->out->data[0] = 0.8; l_test->out->data[1] = 0.2; 

	MSE_derivative_b(Y, l_test);

	printf("Gradient db[0]: %f\n", l_test->db->data[0]);	
	free_layer(l_test);
	free_matrix(Y);
	Y = allocate_matrix(1, 2); 
	Y->data[0] = 1.0; Y->data[1] = 0.0;

	Layer* lt = allocate_layer(2, 1, 2); 

	lt->in->data[0] = 1.0; lt->in->data[1] = 0.0; 
	lt->in->data[2] = 0.0; lt->in->data[3] = 1.0; 
	lt->out->data[0] = 0.5; 
	lt->out->data[1] = 0.5; 

	MSE_derivative_W(Y, lt);

	printf("Test dW[0][0]: %f (Oczekiwane: -0.5)\n", lt->dW->data[0]);
	printf("Test dW[0][1]: %f (Oczekiwane: 0.5)\n", lt->dW->data[1]);

	free_layer(lt);
	free_matrix(Y);
	return 1;
}
void substract_matrices(matrix* mat1, matrix* mat2,matrix* sub_mat)
{

	if ((mat1->rows != mat2->rows) || (mat1->columns != mat2->columns))
	{
		printf("Wrong matrix dimensions!");
		return;
	}
	if ((mat1->rows != sub_mat->rows) || (mat1->columns != sub_mat->columns))
	{
		printf("Wrong matrix dimensions!");
		return;
	}

	float sum = 0;
	for (u32 i =0; i < mat1->rows; i++)
	{
		for (u32 j =0; j < mat2->columns; j++)
		{
			float a = mat1->data[i*mat1->columns +j];
			float b = mat2->data[i*mat2->columns + j];
			sub_mat->data[i*mat1->columns +j] = (float)a - (float)b;
		}
	
	}
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
	l->dz = allocate_matrix(l->num_out,batch_size);
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
	free_matrix(l->dz);
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
	return sum / (double)Y->columns;
}
void hadamard_multiply(matrix* mat1, matrix* mat2,matrix* mul_mat)
{
	for (u32 i = 0; i < mat1->rows*mat2->columns; i++)
	{
		mul_mat->data[i] = mat1->data[i]*mat2->data[i];
	}
}
void fill_matrix_with_value(matrix* mat1, double value)
{
	for(u32 i = 0; i < mat1->rows*mat1->columns;i++)
	{
		mat1->data[i] = value;
	}
}
void scale_matrix(matrix* mat1, double value,matrix* scale_mat)
{
	for(u32 i = 0; i < mat1->rows*mat1->columns; i++)
	{
		scale_mat->data[i] = mat1->data[i]*value;
	}
}
void backpropagation(Network* nn, matrix* Y, float lr) {
	//dz,dW,db for the last layer
	Layer* last = nn->layers[nn->num_layers -1];
	matrix* error = allocate_matrix(Y->rows,Y->columns);
	matrix* ones = allocate_matrix(Y->rows,Y->columns);
	matrix* t_out = allocate_matrix(Y->rows,Y->columns);
	matrix* t_in = allocate_matrix(last->in->columns,last->in->rows);
	transpose_matrix(last->in,t_in);
	substract_matrices(last->out, Y,error);
	fill_matrix_with_value(ones,1);
	substract_matrices(ones,last->out,t_out);
	hadamard_multiply(last->out,t_out,t_out);
	hadamard_multiply(error,t_out,last->dz);
	scale_matrix(last->dz,(2/(double)Y->columns),last->dz);
	multiply_matrices(last->dz,t_in,last->dW);
	for (u32 i = 0; i < last->num_out;i++)
	{
		float sum = 0;
		for (int j =0; j < Y->columns;j++)
		{
			sum += last->dz->data[i*Y->columns + j];
		}
		last->db->data[i] = sum;
	}
	free_matrix(error);
	free_matrix(ones);
	free_matrix(t_out);
	free_matrix(t_in);
	for(i32 i = nn->num_layers -2; i >=0; i--)
	{
		Layer* current = nn->layers[i];
		Layer* next = nn->layers[i+1];
		matrix* W_t = allocate_matrix(next->W->columns,next->W->rows);
		transpose_matrix(next->W,W_t);
     		multiply_matrices(W_t,next->dz,current->dz);
		free_matrix(W_t);
		for(u32 j = 0; j < current->num_out*current->z->columns;j++)
		{
			if (current->z->data[j] > 0)
			{
				continue;
			}
      			else
			{
				current->dz->data[j] = 0;
			}
		}
		matrix* t_in = allocate_matrix(current->in->columns,current->in->rows);
		transpose_matrix(current->in,t_in);
		multiply_matrices(current->dz,t_in,current->dW);
		free_matrix(t_in);
		for (u32 k = 0; k < current->num_out;k++)
		{
			float sum = 0;
			for (int j =0; j < Y->columns;j++)
			{
				sum += current->dz->data[k*Y->columns + j];
			}
			current->db->data[k] = sum;
		}
	}
	for(u32 i = 0; i < nn->num_layers;i++)
	{
		scale_matrix(nn->layers[i]->dW,lr,nn->layers[i]->dW);
		substract_matrices(nn->layers[i]->W,nn->layers[i]->dW,nn->layers[i]->W);
		scale_matrix(nn->layers[i]->db,lr,nn->layers[i]->db);
		substract_matrices(nn->layers[i]->b,nn->layers[i]->db,nn->layers[i]->b);

	}
		//W = W - lr*dW
		//b = b - lr * db
}
i64 time_diff(struct timespec a, struct timespec b)
{
	return ((b.tv_sec - a.tv_sec) * 1000000000LL + (b.tv_nsec - a.tv_nsec));
}
