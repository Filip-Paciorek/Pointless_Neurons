#ifndef NN_H
#define NN_H
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
// type declarations
typedef int8_t i8;
typedef uint8_t u8;
typedef int16_t i16;
typedef uint16_t u16;
typedef int32_t i32;
typedef uint32_t u32;
typedef int64_t i64;
typedef uint64_t u64;
// struct declarations
typedef struct
{
	u32 rows;
	u32 columns;
	float* data;
} matrix;
// function declarations
matrix* allocate_matrix(u32 rows, u32 columns);
void transpose_matrix(matrix* mat,matrix* mat_t);
void multiply_matrices(matrix* mat1, matrix* mat2,matrix* mult_mat);
void add_matrices(matrix* mat1,matrix* mat2, matrix* add_mat);
void add_vector_to_matrix(matrix* mat,matrix* vec,matrix* add_mat);
void free_matrix(matrix* mat);
void print_matrix(matrix* mat);
void fill_matrix(matrix* mat, int start, int end);
i64 time_diff(struct timespec a, struct timespec b);
#endif
