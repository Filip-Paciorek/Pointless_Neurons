#pragma once
#include "nn.h"
typedef struct {
	u32 num_in;
	u32 num_out;
	matrix* in;
	matrix* t_in;
	matrix* W;
	matrix* dW;
	matrix* t_W;
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

void Sigmoid_derivative(Layer* l);
void ReLU_derivative(Layer* l, u32 rows, u32 columns);
void substract_matrices(matrix* mat1, matrix* mat2, matrix* sub_mat);
void copy_to_matrix(matrix* mat1, float* data,float data_rows, float data_columns);
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
void add_bias_to_z(matrix* z, matrix* b);
void He_initialization(Layer* l);
void Xavier_initialization(Layer* l);
void ReLU_activation(Layer* l);
void Sigmoid_activation(Layer* l);
void initialize_network(Network* nn);
//BACKPROP FUNCTIONS
float MSE(matrix* Y, Layer* Y_PRED);
void Gradient_Descent(Layer* l,matrix* Y,u32 learning_rate);
void backpropagation(Network* nn, matrix* Y, float lr,matrix* error, matrix* ones,matrix* t_out,matrix* t_in,matrix** W_ts, matrix** in_ts);
void train(Network* nn, matrix* X, matrix* Y, float lr, u32 epochs, u32 batch_size,float *loss_history);
void test(Network* nn, matrix* X_test, matrix* Y_test);
void feedforward(Network* nn,matrix* X);
matrix* predict(Network* nn, matrix* X);
void load_input(Network* nn, matrix* X);
void print_result(Network* nn, matrix* X_test, matrix* Y_test);
void evaluate(Network *nn, matrix *test_X, matrix *test_Y,float threshold);
