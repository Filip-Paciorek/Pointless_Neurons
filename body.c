#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nn.h"

FILE * File_open(char *file_name){
    FILE *fptr;

    fptr = fopen(file_name,"r");

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
void parse_csv_line(char *line, matrix *X, matrix *Y, u32 row_index){
    const char s[4]=",";
    char *tok;
    tok = strtok(line,s);
    u32 col_index = 0;
    while (tok!=NULL)
    {   
        double val = atof(tok);
        if (col_index<8)
        {
            X->data[((row_index*8)+col_index)] = val;
        }
        else if (col_index==8)
        {
            Y->data[row_index] = val;
        }
        tok = strtok(NULL,s);
        col_index+=1;
        
    }
    
    
}
void standardize_z_score(matrix *mat){

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
    
    }
    }
    
}




