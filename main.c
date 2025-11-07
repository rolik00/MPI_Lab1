#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#define TAG_A 1
#define TAG_B 2
#define TAG_C 3

double** allocate_matrix(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
    }
    return matrix;
}

void free_matrix(double** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void init_matrix(double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX * 10.0;
        }
    }
}
double sequential_multiply(int n) {
    double** A = allocate_matrix(n, n);
    double** B = allocate_matrix(n, n);
    double** C = allocate_matrix(n, n);
    srand(time(NULL));
    init_matrix(A, n, n);
    init_matrix(B, n, n);
    
    double start = MPI_Wtime();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    double end = MPI_Wtime();
    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(C, n);
    return end - start;
}
int main(int argc, char* argv[]) {
    int rank, size;
    int n = 256;
    int q;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    q = (int)sqrt(size);
    if (q * q != size) {
        if (rank == 0) {
            printf("Ошибка с количеством процессов\n");
        }
        MPI_Finalize();
        return 1;
    }
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (n % q != 0) {
        if (rank == 0) {
            printf("Ошибка с размером матрицы\n", q, n);
        }
        MPI_Finalize();
        return 1;
    }
    int block_size = n / q;
    double** local_A = allocate_matrix(block_size, block_size);
    double** local_B = allocate_matrix(block_size, block_size);
    double** local_C = allocate_matrix(block_size, block_size);
    double* send_buffer_A = (double*)malloc(block_size * block_size * sizeof(double));
    double* recv_buffer_A = (double*)malloc(block_size * block_size * sizeof(double));
    double* send_buffer_B = (double*)malloc(block_size * block_size * sizeof(double));
    double* recv_buffer_B = (double*)malloc(block_size * block_size * sizeof(double));
    srand(time(NULL) + rank);
    init_matrix(local_A, block_size, block_size);
    init_matrix(local_B, block_size, block_size);
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            local_C[i][j] = 0.0;
        }
    }
    MPI_Comm grid_comm;
    int dims[2] = {q, q};
    int periods[2] = {1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);
    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int left, right, up, down;
    MPI_Cart_shift(grid_comm, 1, -1, &left, &right);
    MPI_Cart_shift(grid_comm, 0, -1, &up, &down);
    double start_time, end_time;
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    int shift = coords[0];
    for (int step = 0; step < shift; step++) {
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                send_buffer_A[i * block_size + j] = local_A[i][j];
            }
        }
        MPI_Sendrecv(send_buffer_A, block_size * block_size, MPI_DOUBLE, left, TAG_A,
                    recv_buffer_A, block_size * block_size, MPI_DOUBLE, right, TAG_A,
                    grid_comm, MPI_STATUS_IGNORE);
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                local_A[i][j] = recv_buffer_A[i * block_size + j];
            }
        }
    }
    shift = coords[1];
    for (int step = 0; step < shift; step++) {
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                send_buffer_B[i * block_size + j] = local_B[i][j];
            }
        }      
        MPI_Sendrecv(send_buffer_B, block_size * block_size, MPI_DOUBLE, up, TAG_B,
                    recv_buffer_B, block_size * block_size, MPI_DOUBLE, down, TAG_B,
                    grid_comm, MPI_STATUS_IGNORE);
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                local_B[i][j] = recv_buffer_B[i * block_size + j];
            }
        }
    }
    for (int step = 0; step < q; step++) {
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                for (int k = 0; k < block_size; k++) {
                    local_C[i][j] += local_A[i][k] * local_B[k][j];
                }
            }
        }
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                send_buffer_A[i * block_size + j] = local_A[i][j];
            }
        }
        MPI_Sendrecv(send_buffer_A, block_size * block_size, MPI_DOUBLE, left, TAG_A,
                    recv_buffer_A, block_size * block_size, MPI_DOUBLE, right, TAG_A,
                    grid_comm, MPI_STATUS_IGNORE);
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                local_A[i][j] = recv_buffer_A[i * block_size + j];
            }
        }
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                send_buffer_B[i * block_size + j] = local_B[i][j];
            }
        }
        MPI_Sendrecv(send_buffer_B, block_size * block_size, MPI_DOUBLE, up, TAG_B,
                    recv_buffer_B, block_size * block_size, MPI_DOUBLE, down, TAG_B,
                    grid_comm, MPI_STATUS_IGNORE);
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                local_B[i][j] = recv_buffer_B[i * block_size + j];
            }
        }
    }
    end_time = MPI_Wtime();
    double parallel_time = end_time - start_time;
    if (rank == 0) {
        printf("Алгоритм Кэннона\n");
        printf("Размер матрицы: %d x %d\n", n, n);
        printf("Процессы: %d\n", size);
        printf("Параллельное время: %.6f сек\n", parallel_time);
        double seq_time = sequential_multiply(n);
        printf("Последовательное время: %.6f сек\n", seq_time);
        printf("Ускорение: %.2f\n", seq_time / parallel_time);
        printf("Эффективность: %.1f%%\n", (seq_time / parallel_time) / size * 100);
    }
    free(send_buffer_A);
    free(recv_buffer_A);
    free(send_buffer_B);
    free(recv_buffer_B);
    free_matrix(local_A, block_size);
    free_matrix(local_B, block_size);
    free_matrix(local_C, block_size);
    MPI_Comm_free(&grid_comm);
    MPI_Finalize();
    return 0;
}