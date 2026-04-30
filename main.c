#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "nn.h"
#include "brain.c"
#include "body.c"
void shuffle_data(matrix *X, matrix *Y);
int main()
{
    srand(time(NULL));
    //setup memory for data
    matrix *X_raw = allocate_matrix(TRAIN_SIZE, INPUT_COLS);
    matrix *Y_raw = allocate_matrix(TRAIN_SIZE, 1);
    matrix *X_test_raw = allocate_matrix(TEST_SIZE, INPUT_COLS);
    matrix *Y_test_raw = allocate_matrix(TEST_SIZE, 1);

    // Load and Normalize data
    printf("Loading data...\n");
    load_data("pulsary_fixed.csv", X_raw, Y_raw, X_test_raw, Y_test_raw);
    //shuffle the data with an ai function (waiting for maks' implementation)
    //shuffle_data(X_raw, Y_raw);
    //standardize data
    standardize_z_score(X_raw, X_test_raw);
    //second allocation for shuffled data
    matrix *X_train = allocate_matrix(INPUT_COLS, TRAIN_SIZE);
    matrix *Y_train = allocate_matrix(1, TRAIN_SIZE);
    matrix *X_test = allocate_matrix(INPUT_COLS, TEST_SIZE);	
    matrix *Y_test = allocate_matrix(1, TEST_SIZE);
    transpose_matrix(X_raw, X_train);
    transpose_matrix(Y_raw, Y_train);
    transpose_matrix(X_test_raw,X_test);
    transpose_matrix(Y_test_raw,Y_test);
    // Input: 8, Hidden: 16, Output: 1
    u32 layers_config[] = {8, 32, 1};
    //batch size	
    u32 batch_size = 32;
    //create the network struct
    Network* nn = create_network_structure(layers_config, 3, batch_size);

    //Train
    printf("Starting training on Pulsar Dataset...\n");
    train(nn, X_train, Y_train, 2, 5000,batch_size);
    //print results (currently ai function until maks's implementation)
    print_result(nn,X_test,Y_test);
    // freeing
    free_matrix(X_raw); free_matrix(Y_raw);
    free_matrix(X_test_raw); free_matrix(Y_test_raw);
    free_matrix(X_train); free_matrix(Y_train);
    free_matrix(X_test); free_matrix(Y_test);
    free_network(nn);

    return 0;
}
void shuffle_data(matrix *X, matrix *Y) {
    for (u32 i = X->rows - 1; i > 0; i--) {
        u32 j = rand() % (i + 1);
        
        for (u32 k = 0; k < X->columns; k++) {
            float temp = X->data[i * X->columns + k];
            X->data[i * X->columns + k] = X->data[j * X->columns + k];
            X->data[j * X->columns + k] = temp;
        }
        
        float tempY = Y->data[i];
        Y->data[i] = Y->data[j];
        Y->data[j] = tempY;
    }
}
