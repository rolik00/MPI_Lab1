#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#define EPS 0.01
#define MAX_ITER 1000

void input_dim(int rank, int* n, double* c) {
	if (rank == 0) {
		scanf("%d %lf", n, c);
	}

	MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(c, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void generate_u(double* u, int n, double c) {
	srand(time(NULL));

	for (int i = 0; i < n * n; i++) {
		if (i < n || i % n == 0 || i % n == n - 1 || i > n * (n - 1)) u[i] = c;
		else u[i] = ((double)rand() / RAND_MAX) * 200.0 - 100.0;
	}
}

void print_u(double* u, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%7.2f ", u[n * i + j]);
		}
		printf("\n");
	}
}

void create_and_distribute_data(int rank, int n, double c, double* u) {
	if (rank == 0) {
		generate_u(u, n, c);
		print_u(u, n);
	}
	MPI_Bcast(u, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void report_fourth_task(double* u, int n, int iter, double execution_time, int comm_sz, int rank, char* algorithm_name) {
	if (rank == 0) {
		printf("===============================================\n");
		printf("%s\n", algorithm_name);
		printf("===============================================\n");
		printf("Number of processes: %d\n", comm_sz);
		printf("Size of grid: %dx%d\n", n, n);
		print_u(u, n);
		printf("Count of iterations: %d\n", iter);
		printf("Execution time: %.6f seconds\n", execution_time);
		printf("===============================================\n");
	}
}

// Функция для вычисления f(x,y) = -2(x² + y²)
double f(double x, double y) {
	return -2 * (x * x + y * y);
}

// Последовательная версия метода Гаусса-Зейделя
void gauss_seidel_algorithm_sequential(double* u, int n, int* iter) {
	double dmax = 0.0;
	double h = (double)1 / (n - 1);
	do {
		dmax = 0.0;
		for (int i = 1; i < n - 1; i++) {
			for (int j = 1; j < n - 1; j++) {
				double prev = u[i * n + j];
				u[i * n + j] = 0.25 * (u[(i - 1) * n + j] + u[(i + 1) * n + j] + u[i * n + (j - 1)] + u[i * n + (j + 1)] - h * h * f(i * h, j * h));
				double dm = fabs(prev - u[i * n + j]);
				if (dmax < dm) dmax = dm;
			}
		}
		(*iter)++;
	} while (dmax > EPS && *iter < MAX_ITER);
}

// Параллельная версия метода Гаусса-Зейделя
void gauss_seidel_algorithm_parallel(int rank, int comm_sz, double* u, int n, int* iter) {
    double h = (double)1 / (n - 1);
    double dmax, global_dmax;

    int rows_per_proc = (n - 2) / comm_sz;  
    int remainder = (n - 2) % comm_sz;

    int start_row = 1;
    int end_row = 0;

    if (rank < remainder) {
        start_row = 1 + rank * (rows_per_proc + 1);
        end_row = start_row + rows_per_proc;
    } else {
        start_row = 1 + remainder * (rows_per_proc + 1) + (rank - remainder) * rows_per_proc;
        end_row = start_row + rows_per_proc - 1;
    }

    if (rank == comm_sz - 1) {
        end_row = n - 2;
    }

    int local_rows = end_row - start_row + 1;

    do {
        dmax = 0.0;
        if (comm_sz > 1) {
            if (rank > 0) {
                MPI_Sendrecv(&u[start_row * n], n, MPI_DOUBLE, rank - 1, 0, &u[(start_row - 1) * n], n, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            if (rank < comm_sz - 1) {
                MPI_Sendrecv(&u[end_row * n], n, MPI_DOUBLE, rank + 1, 1, &u[(end_row + 1) * n], n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        for (int i = start_row; i <= end_row; i++) {
            for (int j = 1; j < n - 1; j++) {
                double prev = u[i * n + j];
                u[i * n + j] = 0.25 * (u[(i - 1) * n + j] + u[(i + 1) * n + j] + u[i * n + (j - 1)] + u[i * n + (j + 1)] - h * h * f(i * h, j * h));
                double dm = fabs(prev - u[i * n + j]);
                if (dmax < dm) dmax = dm;
            }
        }

        MPI_Allreduce(&dmax, &global_dmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        (*iter)++;

    } while (global_dmax > EPS && *iter < MAX_ITER);

    if (comm_sz> 1) {
        if (rank == 0) {
            for (int p = 1; p < comm_sz; p++) {
                int p_start_row, p_end_row;
                if (p < remainder) {
                    p_start_row = 1 + p * (rows_per_proc + 1);
                    p_end_row = p_start_row + rows_per_proc;
                }
                else {
                    p_start_row = 1 + remainder * (rows_per_proc + 1) + (p - remainder) * rows_per_proc;
                    p_end_row = p_start_row + rows_per_proc - 1;
                }
                if (p == comm_sz - 1) {
                    p_end_row = n - 2;
                }

                int p_rows = p_end_row - p_start_row + 1;
                MPI_Recv(&u[p_start_row * n], p_rows * n, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        else {
            MPI_Send(&u[start_row * n], local_rows * n, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        }
    }
}

void copy_u(double* u, double* u_copy, int n) {
	for (int i = 0; i < n * n; i++) {
		u_copy[i] = u[i];
	}
}

void compare_with_sequential(int rank, int comm_sz, double* u, int n) {
	if (rank == 0) {
		double start_time = MPI_Wtime();

		int iter = 0;
		gauss_seidel_algorithm_sequential(u, n, &iter);

		double end_time = MPI_Wtime();
		double execution_time = end_time - start_time;

		report_fourth_task(u, n, iter, execution_time, comm_sz, rank, "Sequential Gauss-Seidel algorithm");
	}
}

void fourth_task_execute(int my_rank, int comm_sz) {
	int n = 0, iter = 0;
	double c = 0.0;

	input_dim(my_rank, &n, &c);

	double* u = malloc(n * n * sizeof(double));
	create_and_distribute_data(my_rank, n, c, u);

	double* u_copy = malloc(n * n * sizeof(double));
	copy_u(u, u_copy, n);
	compare_with_sequential(my_rank, comm_sz, u_copy, n);

	MPI_Barrier(MPI_COMM_WORLD);
	double start_time = MPI_Wtime();

	gauss_seidel_algorithm_parallel(my_rank, comm_sz, u, n, &iter);

	double end_time = MPI_Wtime();
	double execution_time = end_time - start_time;

	report_fourth_task(u, n, iter, execution_time, comm_sz, my_rank, "Parallel Gauss-Seidel algorithm");
}

void load_data(int rank, int* n, double* c, int value) {
	if (rank == 0) {
		*n = value;
		srand(time(NULL));
		*c = (double)rand() / RAND_MAX * 10.0;
	}

	MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(c, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void fourth_task_report(int my_rank, int comm_sz) {
	int ns[] = { 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000 };
	for(int i = 0; i < 12; i++) {
		int n = 0, iter = 0;
		double c = 0.0;
		load_data(my_rank, &n, &c, ns[i]);

		double* u = malloc(n * n * sizeof(double));
		create_and_distribute_data(my_rank, n, c, u);

		double* u_copy = malloc(n * n * sizeof(double));
		copy_u(u, u_copy, n);
		compare_with_sequential(my_rank, comm_sz, u_copy, n);

		MPI_Barrier(MPI_COMM_WORLD);
		double start_time = MPI_Wtime();

		gauss_seidel_algorithm_parallel(my_rank, comm_sz, u, n, &iter);

		double end_time = MPI_Wtime();
		double execution_time = end_time - start_time;

		report_fourth_task(u, n, iter, execution_time, comm_sz, my_rank, "Parallel Gauss-Seidel algorithm");
	}
}
