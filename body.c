#include <stdio.h>
#include <stdlib.h>
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



