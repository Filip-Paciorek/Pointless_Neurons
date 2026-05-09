#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "nn.h"
#include "brain.h"


FILE * File_open(char *file_name,char *action){
    FILE *fptr;

    fptr = fopen(file_name,action);

    if (fptr == NULL)
    {
        printf("File opening ERROR!\n");
        return NULL;
    }
    return fptr;

}
void file_close(FILE *fptr){
    fclose(fptr);
}
void load_data(char *filename,matrix *X,matrix *Y,matrix* X_t,matrix* Y_t){
    FILE *fptr = File_open(filename,"r");
    char line[1024];
    u32 row_index = 0;
    if (fptr == NULL)
    {
        printf("ERROR");
        return;
    }

    while (fgets(line,sizeof(line),fptr)!=NULL){
         
        if (row_index < TRAIN_SIZE)
        {
            parse_csv_line(line,X,Y,row_index);

        }
        else{
            parse_csv_line(line,X_t,Y_t,row_index-TRAIN_SIZE);
        }
        row_index += 1;}
    file_close(fptr);
}
void parse_csv_line(char *line, matrix *X, matrix *Y, u32 row_index){//function which "cuts" values in lines(from csv) separated , and convert them from string to float value
    const char s[4]=",";//seting char which will will be activator for strtok function to cut values
    char *tok;
    tok = strtok(line,s);// setting token to first line in csv file 
    u32 col_index = 0;
    while (tok!=NULL)
    {   
        double val = atof(tok);//atof convert string values into double
        if (col_index<INPUT_COLS)
        {
            //storing feature of stars in matrix X
            X->data[((row_index*INPUT_COLS)+col_index)] = val;
        }
        else if (col_index==INPUT_COLS)
        {   
            //storing final prediction it if star or not(1 or 0)
            Y->data[(row_index*Y->columns)] = val;
        }
        tok = strtok(NULL,s);//moving to second value in same line
        col_index+=1;
        
    }
}

void standardize_z_score(matrix *mat,matrix *mat_test){

    for (u32 i = 0; i < INPUT_COLS; i++){
    double sum = 0.0;
        for (u32 j = 0; j < mat->rows; j++){
            sum+=mat->data[((j*INPUT_COLS))+i];
        }
    double mean = sum / mat->rows;
    double variation = 0.0;
        for (u32 k = 0; k < mat->rows; k++){
            variation+=pow((mat->data[((k*INPUT_COLS))+i]-mean),2);
        }
    double deviation = sqrt(variation / mat->rows);
    if (deviation>0.0000001){
        for (u32 l = 0; l < mat->rows; l++)
    {
        mat->data[(l*INPUT_COLS)+i] = (mat->data[((l*INPUT_COLS))+i]-mean)/deviation;
    }
        for (u32 m = 0;m < TEST_SIZE; m++)
    {   
            mat_test->data[(INPUT_COLS*m)+i] = (mat_test->data[(INPUT_COLS*m)+i]-mean)/deviation;
    }
    }
}   
}
void normalize_min_max(matrix *mat,matrix *mat_test){
    
    for (u32 i = 0; i < INPUT_COLS; i++){
        double max = mat->data[i];
        double min = mat->data[i];
        for (u32 j = 1; j < mat->rows; j++){
            if (mat->data[(j*INPUT_COLS)+i]>max) max = mat->data[(j*INPUT_COLS)+i];
            if (mat->data[(j*INPUT_COLS)+i]<min) min = mat->data[(j*INPUT_COLS)+i];
        }
        double diff = max - min;
        if (diff >0.0000001){
            for (u32 l = 0; l < mat->rows; l++)
            {
                mat->data[(INPUT_COLS*l)+i] = (mat->data[(INPUT_COLS*l)+i]-min)/diff;
            }
            for (u32 m = 0;m < TEST_SIZE; m++)
            {
                mat_test->data[(INPUT_COLS*m)+i] = (mat_test->data[(INPUT_COLS*m)+i]-min)/diff;
            }
        }

    }
}
void save_to_csv(char *filename,float *arr,u32 size){
    FILE *fptr = File_open(filename,"w");
    for (u32 i = 0; i < size; i++)
    {
        fprintf(fptr,"%.4f\n",arr[i]);
    }
    file_close(fptr);
    
}
void save_matrix_to_csv(char *filename , matrix *mat){
    FILE *fptr = File_open(filename,"w");
    if (mat->rows == 1 ){
        for ( u32 r = 0; r < mat->columns; r++)
        {
            fprintf(fptr,"%.4f\n",mat->data[r]);
        }
    }
    else{

        for (u32 i = 0; i < mat->rows; i++)
    {
        fprintf(fptr,"%.4f",mat->data[(i*mat->columns)+0]);
        for (u32 j = 1; j < mat->columns; j++)
        {
            fprintf(fptr,",%.4f",mat->data[(i*mat->columns)+j]);
        }
        fprintf(fptr,"\n");
        
    }
    }
    file_close(fptr);
}
void shuffle_data(matrix * X,matrix *Y){
    float tmp_X, tmp_Y;
    for (u32 i= X->rows - 1 ;i > 0; i--)
    {
        u32 j = rand() % (i+1);
        
        for (u32 k = 0; k < X->columns; k++)
        {
            tmp_X = X->data[(i*X->columns)+k];
            X->data[(i*X->columns)+k] = X->data[(j*X->columns)+k]; 
            X->data[(j*X->columns)+k] = tmp_X;
        }
        tmp_Y = Y->data[i];
        Y->data[i] = Y->data[j];
        Y->data[j] = tmp_Y;  
    }

}
void evaluate(Network *nn, matrix *test_X, matrix *test_Y){
    u32 TP=0,TN=0,FP=0,FN=0;
    float brier_sum = 0.0f;
    
    matrix *pred_val = predict(nn,test_X);
    for (u32 i = 0; i < test_X->columns; i++)
    {
        if (pred_val->data[i]>=0.5 && test_Y->data[i]==1) TP+=1;
        if (pred_val->data[i]<0.5 && test_Y->data[i]!=1) TN+=1;
        if (pred_val->data[i]>=0.5 && test_Y->data[i]!=1) FP+=1;
        if (pred_val->data[i]<0.5 && test_Y->data[i]==1) FN+=1;
        brier_sum+=(pred_val->data[i]-test_Y->data[i])*(pred_val->data[i]-test_Y->data[i]);
    }
        free_matrix(pred_val); 
    float  Accuracy = (float)(TP + TN) / (float)(TP + TN + FP +FN); 
    float Precision = 0.0f;
    float Recall = 0.0f;
    float F1 = 0.0f;
    float MCC = 0.0f;
    float BrierScore = brier_sum/(float)(test_X->columns);

    
    if (TP + FP) Precision = (float)(TP) / (float)(TP + FP); 
    
    if (TP +  FN) Recall = (float)(TP) / (float)(TP +  FN);
    
    if (Precision + Recall > 0.0f) F1 = 2*(Precision*Recall)/(Precision+Recall);
    

    MCC = ((double)(TP*TN)-(double)(FP*FN))/(sqrt((double)(TP+FP)*(double)(TP+FN)*(double)(TN+FP)*(double)(TN+FN))+1e-7f);
    printf("\n--- Confusion Matrix ---\n");
    printf("Actual \\ Pred |  Pulsar (1)  |   Noise (0)  |\n");
    printf("--------------|--------------|--------------|\n");
    printf("Pulsar (1)    | \033[0;32m%12u\033[0m | \033[0;31m%12u\033[0m | (Recall: %.2f%%)\n", TP, FN, Recall*100);
    printf("Noise  (0)    | \033[0;31m%12u\033[0m | \033[0;32m%12u\033[0m | (Spec:   %.2f%%)\n", FP, TN, (float)TN/(TN+FP+1e-7f)*100);
    printf("\n--- Metrics ---\n");
    printf("Accuracy:   %.2f%%\n", Accuracy * 100);
    printf("Precision:  %.2f%%\n", Precision*100);
    printf("Recall:     %.2f%%\n", Recall*100);
    printf("F1-Score:   %.4f\n",   F1);
    printf("MCC:        %.4f   [-1.0: Najgorzej | 0.0: Losowo | 1.0: Idealnie]\n", MCC);
    printf("Brier Score:%.4f   [ 0: Idealnie   | 0.25: Losowo |  1: Najgorzej]\n", BrierScore);

}
void copy_matrix(matrix *mat,matrix *copy_matrix){
    if ((mat->rows*mat->columns)> (copy_matrix->rows*copy_matrix->columns) || mat->columns != copy_matrix->columns)
    {
        printf("Error, wrong copy_matrix allocation");
        return;
    }
    else{
        for (u32 i = 0; i < (mat->rows)*mat->columns; i++)
    {
        copy_matrix->data[i] = mat->data[i] ;
    }
    
    }
    
}
u32 oversampled_size(matrix *Train_Y){
    u32 noise_count = 0;
    for (u32 i = 0; i < Train_Y->rows; i++)
    {
        if (Train_Y->data[i] == 0) noise_count+=1;
    }
    return noise_count * 2;

}
matrix* Oversampling(matrix *Train_x, matrix *Train_y, matrix *New_Y,u32 size ){
    u32 current_row = Train_y->rows;
    matrix *New_X = allocate_matrix(size,Train_x->columns);
    copy_matrix(Train_x,New_X);
    copy_matrix(Train_y,New_Y);
    while (current_row < size)
    {
        u32 val = rand() % Train_y->rows;
        if (Train_y->data[val] == 1){
            New_Y->data[current_row] = 1;
            for (u32 j = 0; j < Train_x->columns; j++)
            {
                New_X->data[(current_row*Train_x->columns)+j] = Train_x->data[(val*Train_x->columns)+j];
            }
            current_row+=1; 
        }
    }

    return New_X;
}
void data_profiling(matrix *X, matrix *Y){
    u32 pulsars_counter = 0;
    u32 noise_counters = 0;
    double max_tab[8];
    double min_tab[8];
    double mean_tab[8];
    for ( u32 i = 0; i < Y->rows; i++)
    {
        (Y->data[i] == 1) ? pulsars_counter++ : noise_counters ++; 
    }
    for (u32 i = 0; i < X->columns; i++)
    {
        double max = X->data[i];
        double min = X->data[i];
        double sum = 0.0;
        for (u32 j = 0; j < X->rows; j++)
        {   
            sum+=X->data[(j*X->columns)+i];
            if (X->data[(j*X->columns)+i]>max) max = X->data[(j*X->columns)+i];
            if (X->data[(j*X->columns)+i]<min) min = X->data[(j*X->columns)+i];
        }
        double mean = sum / X->rows;
        min_tab[i] = min;
        max_tab[i] = max;
        mean_tab[i] = mean;
    }
    printf("\n================ DATASET PROFILING ================\n");
    for (u32 k = 0; k < X->columns; k+=2)
    {
        printf(" COL |    MIN     |    MAX     |    MEAN    || COL |    MIN     |    MAX     |    MEAN    \n");
        printf("--------------------------------------------||--------------------------------------------\n");
        printf(" [%u] |%10.4f |%10.4f |%13.4f || [%u] |%10.4f |%10.4f |%10.4f\n", k+1, min_tab[k], max_tab[k], mean_tab[k], 
        k+2, min_tab[k+1], max_tab[k+1], mean_tab[k+1]);        
        printf("--------------------------------------------||---------------------------------------------\n");
    }
        printf("Total Samples:  %u\n", Y->rows);
        printf("Pulsars (1):    %u (%.2f%%)\n", pulsars_counter, (float)pulsars_counter / Y->rows * 100.0f);
        printf("Noise (0):      %u (%.2f%%)\n", noise_counters, (float)noise_counters / Y->rows * 100.0f);
    
}








