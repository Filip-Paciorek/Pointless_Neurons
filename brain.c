#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "nn.h"


int main()
{
	matrix* mat = allocate_matrix(2,4);
	fill_matrix(mat,0,3);
	print_matrix(mat);
	printf("\n-------------\n"); 
	matrix* mat_t = transpose_matrix(mat);
	print_matrix(mat_t);
	matrix* mult_mat = multiply_matrices(mat,mat_t);
	printf("\n----------------\n");
	print_matrix(mult_mat);
	free_matrix(mat);
	free_matrix(mat_t);
	free_matrix(mult_mat);
	return 1;
}


i64 time_diff(struct timespec a, struct timespec b)
{
	return ((b.tv_sec - a.tv_sec) * 1000000000LL + (b.tv_nsec - a.tv_nsec));
}

