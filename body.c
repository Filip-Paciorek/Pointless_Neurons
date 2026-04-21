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


