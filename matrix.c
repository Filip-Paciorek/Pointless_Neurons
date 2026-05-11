#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "nn.h"
//--------------------------------Memory allocation, memory freeing----------------------//
matrix* allocate_matrix(u32 rows, u32 columns)
{
/* A function which takes as arguments num of rows and columns and returns
 * a flattened 2D matrix space */
	matrix* mat = malloc(sizeof(*mat));

	if (mat == NULL)
		return NULL;
	mat->rows = rows;
	mat->columns = columns;
	mat->data = malloc(rows*columns*sizeof(*mat->data));
	if (mat->data == NULL)
	{
		free(mat);
		return NULL;
	}
	//Check if the memory was properly allocated
	return mat;
}


void free_matrix(matrix* mat)
{
/* A function freeing the matrix, important for the sake of symmetry and future-proofing*/ 
	free(mat->data);
	free(mat);
};


//---------------------------------Operations on the shape----------------------------------//
void transpose_matrix(matrix* mat,matrix* mat_t)
{
	u32 columns = mat->columns;
	u32 rows = mat->rows;
	for (u32 i = 0; i < rows; i++)
	{
		for (u32 j = 0; j <columns; j++)
		{ 
			u32 original_index = (i*columns) +j;
			u32 transposed_index = (j*rows)+i;
			mat_t->data[transposed_index] = mat->data[original_index];
		}
	}
}


void slice_matrix_columns(matrix* src, matrix* dst, u32 start, u32 size)
{
    /*slices the matrix into smaller parts, allowing for mini-batch SGD*/
    for (u32 r = 0; r < src->rows; r++)
    {
        for (u32 c = 0; c < size; c++)
	{
            dst->data[r * size + c] = src->data[r * src->columns + (start + c)];
    	}
    }
}


//---------------------------------Matrix filling-----------------------------------------//
void fill_matrix_with_value(matrix* mat1, double value)
{
    u32 n = mat1->rows * mat1->columns;
    for (u32 i = 0; i < n; i++)
    {
        mat1->data[i] = (float)value;
    }
}


void fill_matrix(matrix* mat,int start, int end)
{	
/* A function that fills the flattened matrix with numbers in range start end*/ 
	int sum_of_data = mat->rows*mat->columns;
	for (int i = 0; i < sum_of_data;i++)
	{
		(mat->data)[i] = rand() % (end-start) + start;
	}
}


//---------------------------------Matrix math-----------------------------------------//
void add_matrices(matrix* mat1, matrix* mat2,matrix* add_mat)
{

	if ((mat1->rows != mat2->rows) || (mat1->columns != mat2->columns))
	{
		printf("Wrong matrix dimensions!");
		return;
	}
	if ((mat1->rows != add_mat->rows) || (mat1->columns != add_mat->columns))
	{
		printf("Wrong matrix dimensions!");
		return;
	}

	for (u32 i =0; i < mat1->rows; i++)
	{
		for (u32 j =0; j < mat2->columns; j++)
		{
			float a = mat1->data[i*mat1->columns +j];
			float b = mat2->data[i*mat2->columns + j];
			add_mat->data[i*mat1->columns +j] = (float)a + (float)b;
		}
	
	}
}


void substract_matrices(matrix* mat1, matrix* mat2, matrix* sub_mat)
{
    /*Substract mat2 from mat1*/ 
    if ((mat1->rows != mat2->rows) || (mat1->columns != mat2->columns) ||
        (mat1->rows != sub_mat->rows) || (mat1->columns != sub_mat->columns)) 
    {
        printf("Wrong matrix dimensions in substract_matrices!");
        return;
    }
    for (u32 i = 0; i < mat1->rows; i++)
    {
        for (u32 j = 0; j < mat1->columns; j++) 
	{
            float a = mat1->data[i * mat1->columns + j];
            float b = mat2->data[i * mat2->columns + j];
            sub_mat->data[i * mat1->columns + j] = (float)a - (float)b;
	}
    }
}


void multiply_matrices(matrix* mat1, matrix* mat2,matrix* mult_mat)
{

	if (mat1->columns != mat2->rows)
	{
		printf("Wrong matrix dimensions!");
		return;
	}
	float mult_sum = 0;
	for (u32 i =0; i < mat1->rows; i++)
	{
		for (u32 j =0; j < mat2->columns; j++)
		{
			mult_sum = 0;
			for (u32 k = 0; k < mat1->columns; k++)
			{
				float a = mat1->data[i*mat1->columns +k];
				float b = mat2->data[k*mat2->columns + j];
				mult_sum += (float)a * (float)b;
			}
		mult_mat->data[(i*mat2->columns) + j] = mult_sum;
		}
	
	}
}


void scale_matrix(matrix* mat1, double value, matrix* scale_mat)
{
    u32 n = mat1->rows * mat1->columns;
    for (u32 i = 0; i < n; i++)
    {
        scale_mat->data[i] = (float)(mat1->data[i] * value);
    }
}

void hadamard_multiply(matrix* mat1, matrix* mat2, matrix* mul_mat)
{
    u32 n = mat1->rows * mat1->columns;
    for (u32 i = 0; i < n; i++)
    {
        mul_mat->data[i] = mat1->data[i] * mat2->data[i];
    }
}

void add_vector_to_matrix(matrix* mat, matrix* vec, matrix* add_mat)
{
	if(mat->columns != vec->columns)
	{
		printf("Wrong matrix/vector dimensions!");
		return;
	}
	for (u32 i = 0; i < mat->rows;i++)
	{
		for (u32 j = 0; j < mat->columns; j++)
		{
			float a = mat->data[i*mat->columns + j];
			float b = vec->data[j];
			add_mat->data[i*mat->columns +j] = a+b; 
		}
	}
}

//---------------------------------Matrix printing-----------------------------------------//
void print_matrix(matrix* mat)
{
	for (u32 i = 0; i < mat->rows; i++)
	{
		printf("\n");
		for (u32 j = 0; j < mat->columns; j++)
		{
			printf("%f, ", (mat->data)[(i*(mat->columns))+j]);
		}
	}
}
