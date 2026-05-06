#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "nn.h"
#include "brain.h"

//-----------------------------Main four steps of the network--------------------------//

void train(Network* nn, matrix* X, matrix* Y, float lr, u32 epochs, u32 batch_size,float *loss_history)
{
    /*Perform the full forward-backward loop on each batch*/
    //initialize the network
    initialize_network(nn);
    //calculate how many batches we will operate on
    u32 n_batches = X->columns / batch_size;
    //allocate space for the batches
    matrix* X_batch = allocate_matrix(X->rows, batch_size);
    matrix* Y_batch = allocate_matrix(Y->rows, batch_size);    
    matrix* error = allocate_matrix(Y->rows, batch_size);
    matrix* ones  = allocate_matrix(Y->rows, batch_size);
    matrix* t_out = allocate_matrix(Y->rows, batch_size);
    matrix* t_in  = allocate_matrix(batch_size, nn->layers[nn->num_layers-1]->num_in);
    matrix** W_ts = malloc(nn->num_layers*sizeof(matrix*));
    matrix** in_ts = malloc(nn->num_layers*sizeof(matrix*));

    for (u32 i =0; i < nn->num_layers;i++)
    {

        in_ts[i] = allocate_matrix(batch_size,nn->layers[i]->num_in);
        W_ts[i] = allocate_matrix(nn->layers[i]->num_in,nn->layers[i]->num_out);
    }
    //for each epoch iterate over all the batches
    for (u32 e = 0; e < epochs; e++) 
    {
        for (u32 b = 0; b < n_batches; b++) 
        {
            //slice into said batches
            slice_matrix_columns(X, X_batch, b * batch_size, batch_size);
            slice_matrix_columns(Y, Y_batch, b * batch_size, batch_size);
            //do forward-backward motion on them
            feedforward(nn, X_batch);
            backpropagation(nn, Y_batch, lr,error,ones,t_out,t_in,W_ts,in_ts);
        }
        //print loss for each 1000 epochs
        loss_history[e] = MSE(Y_batch,nn->layers[nn->num_layers-1]);//adding loss data needed to plot
        if (e % 1000 == 0)
	{
            printf("epoch %d loss: %f\n", e,loss_history[e]);
	}
    }
    //free memory

    free_matrix(error);
    free_matrix(ones);
    free_matrix(t_out);
    free_matrix(t_in);
    for (u32 i = 0; i < nn->num_layers;i++)
    {
        free_matrix(W_ts[i]);
        free_matrix(in_ts[i]);
    }
    free(W_ts);
    free(in_ts);
    free_matrix(X_batch);
    free_matrix(Y_batch);
}

void feedforward(Network* nn, matrix* X)
{
    /*Populates the network going through it for the first time*/
    //make sure X doesnt get redefined
    matrix* current_input = X;

    for (int i = 0; i < nn->num_layers; i++) 
    {
        //set the current input as our in for this layer
        nn->layers[i]->in = current_input;
        // calculate z
        calculate_layer(nn->layers[i], current_input);
        //check if it should be ReLU or Sigmoid BUT we switched it to just sigmoid because it performed better
        if (i < nn->num_layers - 1)
        {
            Sigmoid_activation(nn->layers[i]);
        }
        else
    {
            Sigmoid_activation(nn->layers[i]);
        }
        //after activation set the output as our new input
        current_input = nn->layers[i]->out;
    }
}


void backpropagation(Network* nn, matrix* Y, float lr,matrix* error, matrix* ones,matrix* t_out,matrix* t_in,matrix** W_ts, matrix** in_ts)
{
    Layer* last = nn->layers[nn->num_layers - 1];
    u32 batch_size = Y->columns;

    //reset gradients and set up space
    for (u32 i = 0; i < nn->num_layers; i++) 
    {

        transpose_matrix(nn->layers[i]->W, W_ts[i]);
        transpose_matrix(nn->layers[i]->in, in_ts[i]);
        fill_matrix_with_value(nn->layers[i]->dW, 0.0);
        fill_matrix_with_value(nn->layers[i]->db, 0.0);
    }
  //calculate the first derivatives
    //calculate the derivative of simgoid and dz
    transpose_matrix(last->in, t_in); //in^T
    substract_matrices(last->out, Y, error); //out - Y = error
    fill_matrix_with_value(ones, 1.0); 
    substract_matrices(ones, last->out, t_out); // 1 - out = t_out
    hadamard_multiply(last->out, t_out, t_out); // out o (1-out) = t_out
    hadamard_multiply(error, t_out, last->dz); // (out - Y) o (out o (1-out)) = dz
    scale_matrix(last->dz, 2.0 / batch_size, last->dz); //dz * (2/batch_size) =dz

    //calculate derivatives of dw and db for the last layer
    multiply_matrices(last->dz, t_in, last->dW); //dz * in^T = dW
    for (u32 i = 0; i < last->num_out; i++) 
    {
        float sum = 0;
        for (u32 j = 0; j < batch_size; j++)
        {
            sum += last->dz->data[i * batch_size + j]; 
        }
        last->db->data[i] = sum; //dz (sum) = db
    }

  //calculate other derivatives
    //hidden layers
    for (i32 i = nn->num_layers - 2; i >= 0; i--) 
    {
        Layer* current = nn->layers[i];
        Layer* next    = nn->layers[i + 1];

        //propagate dz back
      //calculate the new dz by using the chain rule which is in this case the derivative transposed W *dz * sigmoid derivative;
        multiply_matrices(W_ts[i+1], next->dz, current->dz);
        //multiply by sigmoid'
        Sigmoid_derivative(current);
        //multiply by input transposed to get the current dW
        multiply_matrices(current->dz, in_ts[i], current->dW);
        //get the b derivative by summing the current dz since dz*db = dz * (aW + b)' = dz * 1
        for (u32 k = 0; k < current->num_out; k++) {
            float sum = 0;
            for (u32 j = 0; j < batch_size; j++)
                sum += current->dz->data[k * batch_size + j];
            current->db->data[k] = sum;
        }
    }
    //update the weights
    for (u32 i = 0; i < nn->num_layers; i++) 
    {
        Layer* l = nn->layers[i];
        for (u32 j = 0; j < l->W->rows * l->W->columns; j++)
            l->W->data[j] -= lr * l->dW->data[j];
        for (u32 j = 0; j < l->b->rows; j++)
            l->b->data[j] -= lr * l->db->data[j];
    }
}


matrix* predict(Network* nn, matrix* X)
{
    //make sure the X doesnt get changed 
    matrix* current_input = X;
    //matrix* current_output = NULL;

    //allocate space for z and out for each layer sized to X->columns
    //matrix** temps = malloc(nn->num_layers * 2 * sizeof(matrix*));

    for (int i = 0; i < nn->num_layers; i++) 
    {
        //allocate memory for the temp z and out
        Layer* l = nn->layers[i];
        matrix* z_temp   = allocate_matrix(l->num_out, X->columns);
        matrix* out_temp = allocate_matrix(l->num_out, X->columns);
        //calculate z 
        multiply_matrices(l->W, current_input, z_temp);
        add_bias_to_z(z_temp, l->b);
        //calculate the sigmoid
        u32 n = z_temp->rows * z_temp->columns;
        for (u32 j = 0; j < n; j++)
        {
            out_temp->data[j] = 1.0f / (1.0f + expf(-z_temp->data[j]));
        }
        //free temp
        free_matrix(z_temp);
        //free previous out_temp
        if (current_input != X)
        {
            free_matrix(current_input);
        } 
        //append to temps and update the current input and output
        current_input = out_temp;
    }
    //CALLER MUST FREE THIS!!!
    return current_input;
}
//-----------------------------Math functions for parts of calculations--------------------------//
void add_bias_to_z(matrix* z, matrix* b)
{
    for (u32 i = 0; i < z->columns; i++)
    {
        for (u32 j = 0; j < z->rows; j++)
        {
            z->data[j * z->columns + i] += b->data[j];
        }
    }
}

void calculate_layer(Layer* l, matrix* input)
{
    //calculate z 
    multiply_matrices(l->W, input, l->z);
    add_bias_to_z(l->z, l->b);
}

void ReLU_activation(Layer* l)
{
    u32 all = l->z->rows * l->z->columns;
    for (int i = 0; i < all; i++)
    {
        l->out->data[i] = l->z->data[i] > 0 ? l->z->data[i] : 0;
    }
}

void Sigmoid_activation(Layer* l)
{
    u32 all = l->z->rows * l->z->columns;
    for (int i = 0; i < all; i++)
    {
        l->out->data[i] = 1.0f / (1.0f + expf(-l->z->data[i]));
    }
}


float MSE(matrix* Y, Layer* Y_PRED)
{
    /*Calculates the squared error between the predicted value and the real value*/
    if (Y->rows != Y_PRED->out->rows || Y->columns != Y_PRED->out->columns)
    {
        return -1;
    }
    double sum = 0.0;
    u32 n = Y->rows * Y->columns;
    for (u32 i = 0; i < n; i++) 
    {
        double diff = Y_PRED->out->data[i] - Y->data[i];
        sum += diff * diff;
    }
    return (float)(sum / Y->columns);
}



void Sigmoid_derivative(Layer* l)
{
    u32 n = l->out->rows * l->out->columns;
    for (u32 i = 0; i < n; i++) 
    {
        float a = l->out->data[i];
        l->dz->data[i] *= a * (1.0f - a);
    }
}
//------------------------------Network initialization--------------------------//
void initialize_network(Network* nn)
{
    /*Fills up the initialized structure of weights and bias*/
    for (int i = 0; i < nn->num_layers; i++) 
    {
        Xavier_initialization(nn->layers[i]);
        b_initialization(nn->layers[i]);
    }
}
void b_initialization(Layer* l)
{
    /* Fills up the b param with 0*/
    fill_matrix_with_value(l->b, 0.0);
}

void He_initialization(Layer* l)
{
    double limit = sqrt(2.0 / l->num_in);
    for (int i = 0; i < l->W->rows; i++)
    {
        for (int j = 0; j < l->W->columns; j++)
        {
            l->W->data[i * l->W->columns + j] = (((double)rand() / RAND_MAX) * 2 * limit) - limit;
        }
    }
}

void Xavier_initialization(Layer* l)
{
    float limit = sqrtf(6.0f / (l->num_in + l->num_out));
    for (int i = 0; i < l->W->rows; i++)
    {
        for (int j = 0; j < l->W->columns; j++)
        {
            l->W->data[i * l->W->columns + j] = (((double)rand() / RAND_MAX) * 2 * limit) - limit;
        }
    }
}
//------------------------------Memory allocation and structure creation--------------------------//
Network* create_network_structure(u32* arr, u32 num_layers, u32 batch_size)
{
    /* Creates empty network structure */
    //allocate space for network pointer
    Network* nn = malloc(sizeof(*nn));
    nn->num_layers = num_layers - 1;
    //for each layer pointer allocate space
    nn->layers = malloc(nn->num_layers * sizeof(Layer*));
    //for each layer allocate space
    for (int i = 0; i < nn->num_layers; i++)
    {
        nn->layers[i] = allocate_layer(arr[i], arr[i+1], batch_size);
    }
    return nn;
}
Layer* allocate_layer(u32 num_in, u32 num_out, u32 batch_size)
{
    /*Allocate memory for all the parts owned by the layer*/
    Layer* l = malloc(sizeof(*l));
    l->num_in  = num_in;
    l->num_out = num_out;
    //NOT owned by layer
    l->in  = NULL; 
    l->W   = allocate_matrix(num_out, num_in);
    l->dW  = allocate_matrix(num_out, num_in);
    l->b   = allocate_matrix(num_out, 1);
    l->db  = allocate_matrix(num_out, 1);
    l->z   = allocate_matrix(num_out, batch_size);
    l->dz  = allocate_matrix(num_out, batch_size);
    l->out = allocate_matrix(num_out, batch_size);
    return l;
}
//------------------------------Memory release--------------------------//
void free_network(Network* nn)
{
    /* Frees all the memory allocated for network*/
    for (int i = 0; i < nn->num_layers; i++)
    {
        free_layer(nn->layers[i]);
    }
    free(nn->layers);
    free(nn);
}
void free_layer(Layer* l)
{
    // free every part owned by the layer
    free_matrix(l->W);
    free_matrix(l->dW);
    free_matrix(l->b);
    free_matrix(l->db);
    free_matrix(l->z);
    free_matrix(l->dz);
    free_matrix(l->out);
    free(l);
}
//------------------------------Result printing and debug printing--------------------------//
void print_result(Network* nn, matrix* X_test, matrix* Y_test)
{
    /*function written by ai to test out results (waiting for maks's implementation)*/
    matrix* out = predict(nn, X_test);

    int tp = 0, tn = 0, fp = 0, fn = 0;
    for (u32 i = 0; i < X_test->columns; i++) 
    {
        int predicted = (out->data[i] >= 0.5f) ? 1 : 0;
        int actual    = (int)Y_test->data[i];
        if(predicted == 1 && actual == 1)
        {
            tp++;
        }
        else if(predicted == 0 && actual == 0)
        {
            tn++;
        }
        else if(predicted == 1 && actual == 0)
        {
            fp++;
        }
        else if(predicted == 0 && actual == 1)
        {
            fn++;
        }
    }
    free_matrix(out);

    float precision = (float)tp / (tp + fp + 1e-7f);
    float recall    = (float)tp / (tp + fn + 1e-7f);
    float f1        = 2.0f * precision * recall / (precision + recall + 1e-7f);

    printf("\n--- Confusion Matrix ---\n");
    printf("Actual \\ Pred |  Pulsar (1)  |   Noise (0)  |\n");
    printf("--------------|--------------|--------------|\n");
    printf("Pulsar (1)    | %12d | %12d | (Recall: %.2f%%)\n", tp, fn, recall*100);
    printf("Noise  (0)    | %12d | %12d | (Spec:   %.2f%%)\n", fp, tn, (float)tn/(tn+fp+1e-7f)*100);
    printf("\n--- Metrics ---\n");
    printf("Accuracy:  %.2f%%\n", (float)(tp+tn) / X_test->columns * 100);
    printf("Precision: %.2f%%\n", precision*100);
    printf("Recall:    %.2f%%\n", recall*100);
    printf("F1-Score:  %.4f\n",   f1);
}

void print_layer_weights(Layer* l)
{
    /*Prints layer weights*/
    for (int i = 0; i < l->W->rows; i++) 
    {
        for (int j = 0; j < l->W->columns; j++)
        {
            printf("%f ", l->W->data[i * l->W->columns + j]);
            printf("\n");
        }
    }
}

void print_network_params(Network* nn)
{
    for (int i = 0; i < nn->num_layers; i++)
    {
        printf("Layer %d - in: %d out: %d\n", i, nn->layers[i]->num_in, nn->layers[i]->num_out);
    }
}
void print_layer_outputs(Layer* l)
{
    for (int i = 0; i < l->num_out; i++)
    {
        printf("%f ", l->z->data[i]);
    }
    printf("\n");
}
//------------------------------Other--------------------------//
i64 time_diff(struct timespec a, struct timespec b)
{
    return ((b.tv_sec - a.tv_sec) * 1000000000LL + (b.tv_nsec - a.tv_nsec));
}
