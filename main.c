#include "brain.c"
#include "body.c"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "nn.h"
#include "brain.h"
// Defining epochs and threshold to avoid mistakes 
#define epochs  10000
#define threshold 0.7
int main()
{
    srand(time(NULL));
    // Loading original raw data
    matrix *X_raw = allocate_matrix(TRAIN_SIZE, INPUT_COLS);
    matrix *Y_raw = allocate_matrix(TRAIN_SIZE, 1);
    matrix *X_test_raw = allocate_matrix(TEST_SIZE, INPUT_COLS);
    matrix *Y_test_raw = allocate_matrix(TEST_SIZE, 1);

    printf("Loading and shuffling data...\n");
    load_data("pulsary_fixed.csv", X_raw, Y_raw, X_test_raw, Y_test_raw);
    
    // Shuffling data before cloning to ensure random distribution
    shuffle_data(X_raw, Y_raw); 

    // Checking original dataset statistics before applying oversampling
    data_profiling(X_raw, Y_raw); 


    // DATA BALANCING: OVERSAMPLING
    printf("\nApplying Oversampling to training data...\n");
    
    // Calculating target size based on the noise
    u32 target_size = oversampled_size(Y_raw); 
    
    matrix *Y_balanced = allocate_matrix(target_size, 1);
    matrix *X_balanced = Oversampling(X_raw, Y_raw, Y_balanced, target_size);
    
    // Shuffling the balanced dataset to prevent the network from learning sequences
    shuffle_data(X_balanced, Y_balanced); 
    
    // Releasing original matrices 
    free_matrix(X_raw); 
    free_matrix(Y_raw);
    
    // Saving raw balanced features for Python 
    save_matrix_to_csv("x_raw.csv", X_balanced);

    // WORKSPACE MATRICES
    // Allocating workspace matrices using the new target_size
    matrix *X_work = allocate_matrix(target_size, INPUT_COLS);
    matrix *X_test_work = allocate_matrix(TEST_SIZE, INPUT_COLS); 

    //Transposed dimensions for dot products
    matrix *X_train = allocate_matrix(INPUT_COLS, target_size);
    matrix *Y_train = allocate_matrix(1, target_size);
    matrix *X_test = allocate_matrix(INPUT_COLS, TEST_SIZE);    
    matrix *Y_test = allocate_matrix(1, TEST_SIZE);
    
    // Transposing labels to match network architecture requirements
    transpose_matrix(Y_balanced, Y_train);
    transpose_matrix(Y_test_raw, Y_test);

    // Saving true test labels once to serve as ground truth for Python metrics
    save_matrix_to_csv("y_true.csv", Y_test);

    // Network structural configuration
    u32 layers_config[] = {8, 16, 1};
    u32 batch_size = 32;
    Network* nn = create_network_structure(layers_config, 3, batch_size);



    //Z-SCORE (OVERSAMPLED DATA)
    printf("\n>>> STARTING TRAINING: Z-SCORE + OVERSAMPLING <<<\n");
    
    // Resetting workspace with a fresh copy of balanced data
    copy_matrix(X_balanced, X_work);
    copy_matrix(X_test_raw, X_test_work);
    
    // Applying scaling
    standardize_z_score(X_work, X_test_work);
    
    // Saving scaled features for Python EDA comparison
    save_matrix_to_csv("x_zscore.csv", X_work);
    
    transpose_matrix(X_work, X_train);
    transpose_matrix(X_test_work, X_test);

    float loss_history_zscore[epochs];
    train(nn, X_train, Y_train, 0.1, epochs, batch_size, loss_history_zscore);
    
    // Exporting loss array for Python learning curve plots
    save_to_csv("loss_zscore.csv", loss_history_zscore, epochs);
    
    evaluate(nn, X_test, Y_test, threshold);
    
    printf("\nExporting Z-Score predictions to CSV...\n");
    matrix *final_preds_zscore = predict(nn, X_test);
    
    // Saving raw probability outputs for Python Confusion Matrix and ROC plots
    save_matrix_to_csv("y_pred_zscore.csv", final_preds_zscore);
    free_matrix(final_preds_zscore);

    //MIN-MAX (OVERSAMPLED DATA)

    printf("\n>>> STARTING TRAINING: MIN-MAX + OVERSAMPLING <<<\n");
    
    // Resetting workspace with a fresh copy of balanced data
    copy_matrix(X_balanced, X_work);
    copy_matrix(X_test_raw, X_test_work);
    
    // Applying scaling
    normalize_min_max(X_work, X_test_work);
    
    // Saving scaled features for Python EDA comparison
    save_matrix_to_csv("x_minmax.csv", X_work);
    
    transpose_matrix(X_work, X_train);
    transpose_matrix(X_test_work, X_test);

    float loss_history_minmax[epochs];
    train(nn, X_train, Y_train, 0.1, epochs, batch_size, loss_history_minmax);
    
    // Exporting loss array for Python learning curve plots
    save_to_csv("loss_minmax.csv", loss_history_minmax, epochs);

    evaluate(nn, X_test, Y_test, threshold);
    
    printf("\nExporting Min-Max predictions to CSV...\n");
    matrix *final_preds_minmax = predict(nn, X_test);
    
    // Saving raw probability outputs for Python Confusion Matrix and ROC plots
    save_matrix_to_csv("y_pred_minmax.csv", final_preds_minmax);
    free_matrix(final_preds_minmax);


    // MEMORY CLEANUP
    free_matrix(X_balanced); free_matrix(Y_balanced);
    free_matrix(X_test_raw); free_matrix(Y_test_raw);
    free_matrix(X_work); free_matrix(X_test_work);
    free_matrix(X_train); free_matrix(Y_train);
    free_matrix(X_test); free_matrix(Y_test);
    free_network(nn);

    return 0;
}