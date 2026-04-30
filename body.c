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
void parse_csv_line(char *line, matrix *X, matrix *Y, u32 row_index){
    const char s[4]=",";
    char *tok;
    tok = strtok(line,s);
    u32 col_index = 0;
    while (tok!=NULL)
    {   
        double val = atof(tok);
        if (col_index<INPUT_COLS)
        {
            X->data[((row_index*INPUT_COLS)+col_index)] = val;
        }
        else if (col_index==INPUT_COLS)
        {
            Y->data[(row_index*Y->columns)] = val;
        }
        tok = strtok(NULL,s);
        col_index+=1;
        
    }
}

void standardize_z_score(matrix *mat,matrix *mat_test){

    for (u32 i = 0; i < INPUT_COLS; i++){
    double sum = 0.0;
        for (u32 j = 0; j < TRAIN_SIZE; j++){
            sum+=mat->data[((j*INPUT_COLS))+i];
        }
    double mean = sum / TRAIN_SIZE;
    double variation = 0.0;
        for (u32 k = 0; k < TRAIN_SIZE; k++){
            variation+=pow((mat->data[((k*INPUT_COLS))+i]-mean),2);
        }
    double deviation = sqrt(variation / TRAIN_SIZE);
    if (deviation>0.0000001){
        for (u32 l = 0; l < TRAIN_SIZE; l++)
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
        for (u32 j = 1; j < TRAIN_SIZE; j++){
            if (mat->data[(j*INPUT_COLS)+i]>max) max = mat->data[(j*INPUT_COLS)+i];
            if (mat->data[(j*INPUT_COLS)+i]<min) min = mat->data[(j*INPUT_COLS)+i];
        }
        double diff = max - min;
        if (diff >0.0000001){
            for (u32 l = 0; l < TRAIN_SIZE; l++)
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
void save_matrix_to_csv(char *filename , matrix *mat){
    FILE *fptr = File_open(filename,"w");
    for (u32 i = 0; i < mat->rows; i++)
    {
        fprintf(fptr,"%.4f",mat->data[(i*mat->columns)+0]);
        for (u32 j = 1; j < mat->columns; j++)
        {
            fprintf(fptr,",%.4f",mat->data[(i*mat->columns)+j]);
        }
        fprintf(fptr,"\n");
        
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





