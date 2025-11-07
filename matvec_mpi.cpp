#include <mpi.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>

// === Утилиты ===
std::vector<int> parse_sizes(const std::string &s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try { int v = std::stoi(token); if (v > 0) out.push_back(v); } catch(...) {}
    }
    return out;
}

double pseudo_rand(int i, int j, int n) {
    long long h = (long long)(i + 1) * 1000003LL + (j + 1) * 9176LL + n * 31;
    h ^= (h << 13); h ^= (h >> 7); h ^= (h << 17);
    return ((h % 2000003LL) / 1000003.0) - 1.0;
}

std::vector<double> gen_matrix(int n) {
    std::vector<double> A((size_t)n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[(size_t)i * n + j] = pseudo_rand(i, j, n);
    return A;
}

std::vector<double> gen_vector(int n) {
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i) v[i] = pseudo_rand(i, i, n);
    return v;
}

// === Последовательное умножение ===
double seq_matvec(int n, const std::vector<double> &A, const std::vector<double> &x,
                  std::vector<double> &y) {
    y.resize(n);
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const double *row = &A[(size_t)i * n];
        for (int j = 0; j < n; ++j) s += row[j] * x[j];
        y[i] = s;
    }
    return 0.0;
}

// === Row-wise ===
double matvec_rowwise(int n, const std::vector<double> &A_root, const std::vector<double> &x_root,
                      std::vector<double> &y_out, MPI_Comm comm, int repeats) {
    int rank, procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &procs);

    std::vector<int> rows_per_proc(procs), row_offset(procs);
    int base = n / procs, rem = n % procs;
    for (int p = 0; p < procs; ++p) {
        rows_per_proc[p] = base + (p < rem);
        row_offset[p] = (p == 0 ? 0 : row_offset[p-1] + rows_per_proc[p-1]);
    }

    int local_rows = rows_per_proc[rank];
    std::vector<double> A_local((size_t)local_rows * n);

    std::vector<int> sendcounts(procs), displs(procs);
    for (int p = 0; p < procs; ++p) {
        sendcounts[p] = rows_per_proc[p] * n;
        displs[p] = row_offset[p] * n;
    }

    MPI_Scatterv(rank == 0 ? (void*)A_root.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                 A_local.data(), local_rows * n, MPI_DOUBLE, 0, comm);

    std::vector<double> x(n);
    if (rank == 0) x = x_root;
    MPI_Bcast(x.data(), n, MPI_DOUBLE, 0, comm);

    std::vector<double> y_local(local_rows);

    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    for (int rep = 0; rep < repeats; ++rep) {
        for (int i = 0; i < local_rows; ++i) {
            double s = 0.0;
            const double *row = &A_local[(size_t)i * n];
            for (int j = 0; j < n; ++j) s += row[j] * x[j];
            y_local[i] = s;
        }
    }
    
    double t1 = MPI_Wtime();

    if (rank == 0) y_out.assign(n, 0.0);
    MPI_Gatherv(y_local.data(), local_rows, MPI_DOUBLE,
                y_out.data(), rows_per_proc.data(), row_offset.data(), MPI_DOUBLE, 0, comm);

    double elapsed = (t1 - t0) / repeats;
    return elapsed;
}

// === Col-wise ===
double matvec_colwise(int n, const std::vector<double> &A_root, const std::vector<double> &x_root,
                      std::vector<double> &y_out, MPI_Comm comm, int repeats) {
    int rank, procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &procs);

    std::vector<int> cols_per_proc(procs), col_offset(procs);
    int base = n / procs, rem = n % procs;
    for (int p = 0; p < procs; ++p) {
        cols_per_proc[p] = base + (p < rem);
        col_offset[p] = (p == 0 ? 0 : col_offset[p-1] + cols_per_proc[p-1]);
    }

    int local_cols = cols_per_proc[rank];
    std::vector<double> A_local((size_t)n * local_cols);
    std::vector<double> x_local(local_cols);
    std::vector<double> y_partial(n, 0.0);

    // Подготовка sendbuf на root
    std::vector<double> sendbuf_A;
    std::vector<int> sendcounts_A(procs), displs_A(procs);
    if (rank == 0) {
        sendbuf_A.resize((size_t)n * n);
        int pos = 0;
        for (int p = 0; p < procs; ++p) {
            sendcounts_A[p] = n * cols_per_proc[p];
            displs_A[p] = pos;
            for (int c = 0; c < cols_per_proc[p]; ++c) {
                int col = col_offset[p] + c;
                for (int r = 0; r < n; ++r)
                    sendbuf_A[pos++] = A_root[(size_t)r * n + col];
            }
        }
    } else {
        std::fill(sendcounts_A.begin(), sendcounts_A.end(), 0);
        std::fill(displs_A.begin(), displs_A.end(), 0);
    }

    MPI_Bcast(sendcounts_A.data(), procs, MPI_INT, 0, comm);
    MPI_Bcast(displs_A.data(), procs, MPI_INT, 0, comm);

    MPI_Scatterv(rank == 0 ? sendbuf_A.data() : nullptr, sendcounts_A.data(), displs_A.data(), MPI_DOUBLE,
                 A_local.data(), n * local_cols, MPI_DOUBLE, 0, comm);

    MPI_Scatterv(rank == 0 ? (void*)x_root.data() : nullptr, cols_per_proc.data(), col_offset.data(), MPI_DOUBLE,
                 x_local.data(), local_cols, MPI_DOUBLE, 0, comm);

    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    for (int rep = 0; rep < repeats; ++rep) {
        std::fill(y_partial.begin(), y_partial.end(), 0.0);
        for (int c = 0; c < local_cols; ++c) {
            double xc = x_local[c];
            const double *col = &A_local[(size_t)c * n];
            for (int r = 0; r < n; ++r) y_partial[r] += col[r] * xc;
        }
    }
    
    double t1 = MPI_Wtime();

    if (rank == 0) y_out.assign(n, 0.0);
    MPI_Reduce(y_partial.data(), y_out.data(), n, MPI_DOUBLE, MPI_SUM, 0, comm);

    double elapsed = (t1 - t0) / repeats;
    return elapsed;
}

// === Block-wise (2D decomposition) ===
double matvec_block(int n, const std::vector<double> &A_root, const std::vector<double> &x_root,
                    std::vector<double> &y_out, MPI_Comm comm, int repeats) {
    int rank, procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &procs);

    int P = 1, Q = 1;
    for (int p = (int)std::sqrt(procs); p >= 1; --p) {
        if (procs % p == 0) { P = p; Q = procs / p; break; }
    }
    if (P * Q != procs && rank == 0) {
        std::cerr << "Warning: Cannot form 2D grid, falling back to row-wise\n";
        return matvec_rowwise(n, A_root, x_root, y_out, comm, repeats);
    }

    int row_p = rank / Q;
    int col_p = rank % Q;

    std::vector<int> row_start(P), row_count(P);
    int base_r = n / P, rem_r = n % P;
    for (int p = 0; p < P; ++p) {
        row_start[p] = p * base_r + std::min(p, rem_r);
        row_count[p] = base_r + (p < rem_r);
    }

    std::vector<int> col_start(Q), col_count(Q);
    int base_c = n / Q, rem_c = n % Q;
    for (int q = 0; q < Q; ++q) {
        col_start[q] = q * base_c + std::min(q, rem_c);
        col_count[q] = base_c + (q < rem_c);
    }

    int local_rows = row_count[row_p];
    int local_cols = col_count[col_p];
    int local_size = local_rows * local_cols;

    std::vector<double> sendbuf;
    std::vector<int> sendcounts(procs, 0), displs(procs, 0);
    if (rank == 0) {
        sendbuf.reserve((size_t)n * n);
        int pos = 0;
        for (int rp = 0; rp < P; ++rp) {
            for (int cq = 0; cq < Q; ++cq) {
                int p = rp * Q + cq;
                int r0 = row_start[rp], c0 = col_start[cq];
                int nr = row_count[rp], nc = col_count[cq];
                sendcounts[p] = nr * nc;
                displs[p] = pos;
                for (int i = 0; i < nr; ++i)
                    for (int j = 0; j < nc; ++j)
                        sendbuf.push_back(A_root[(r0 + i) * n + (c0 + j)]);
                pos += nr * nc;
            }
        }
    }

    MPI_Bcast(sendcounts.data(), procs, MPI_INT, 0, comm);
    MPI_Bcast(displs.data(), procs, MPI_INT, 0, comm);

    std::vector<double> A_local(local_size);
    MPI_Scatterv(rank == 0 ? sendbuf.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                 A_local.data(), local_size, MPI_DOUBLE, 0, comm);

    std::vector<double> x(n);
    if (rank == 0) x = x_root;
    MPI_Bcast(x.data(), n, MPI_DOUBLE, 0, comm);

    std::vector<double> y_local_full(n, 0.0);
    int row_start_local = row_start[row_p];

    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    for (int rep = 0; rep < repeats; ++rep) {
        std::fill(y_local_full.begin(), y_local_full.end(), 0.0);
        int c0 = col_start[col_p];
        for (int i = 0; i < local_rows; ++i) {
            double s = 0.0;
            for (int j = 0; j < local_cols; ++j) {
                s += A_local[i * local_cols + j] * x[c0 + j];
            }
            y_local_full[row_start_local + i] = s;
        }
    }
    
    double t1 = MPI_Wtime();

    if (rank == 0) y_out.assign(n, 0.0);
    MPI_Reduce(y_local_full.data(), y_out.data(), n, MPI_DOUBLE, MPI_SUM, 0, comm);

    double elapsed = (t1 - t0) / repeats;
    return elapsed;
}

// === Main ===
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    std::vector<int> sizes = {500, 1000, 2000};
    int base_repeats = 5;
    if (argc >= 2) sizes = parse_sizes(argv[1]);
    if (argc >= 3) { try { base_repeats = std::max(1, std::stoi(argv[2])); } catch(...) {} }

    if (rank == 0) {
        std::cout << "# algorithm,n,procs,T_parallel_sec,T_seq_sec\n";
        std::cout << "# Processes: " << procs << ", base_repeats: " << base_repeats << "\n";
    }

    for (int n : sizes) {
        int repeats = base_repeats;
        if (rank == 0) {
            if (n <= 500) repeats = 100;
            else if (n <= 2000) repeats = 20;
            else if (n <= 5000) repeats = 5;
            else repeats = 1;
        }
        MPI_Bcast(&repeats, 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<double> A_root, x_root, y_seq;
        if (rank == 0) {
            A_root = gen_matrix(n);
            x_root = gen_vector(n);
        }

        // === Эталонное последовательное вычисление ===
        double T1 = 0.0;
        if (rank == 0) {
            double t0 = MPI_Wtime();
            for (int r = 0; r < repeats; ++r)
                seq_matvec(n, A_root, x_root, y_seq);
            double t1 = MPI_Wtime();
            T1 = (t1 - t0) / repeats;
        }
        MPI_Bcast(&T1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        std::vector<double> y_par;

        // === Row-wise ===
        double Trow = matvec_rowwise(n, A_root, x_root, y_par, MPI_COMM_WORLD, repeats);
        if (rank == 0) {
            double max_err = 0.0;
            for (int i = 0; i < n; ++i)
                max_err = std::max(max_err, std::abs(y_par[i] - y_seq[i]));
            std::cout << std::fixed << std::setprecision(9)
                      << "row," << n << "," << procs << ","
                      << Trow << "," << T1
                      << ", error=" << max_err << "\n";
        }

        // === Col-wise ===
        double Tcol = matvec_colwise(n, A_root, x_root, y_par, MPI_COMM_WORLD, repeats);
        if (rank == 0) {
            double max_err = 0.0;
            for (int i = 0; i < n; ++i)
                max_err = std::max(max_err, std::abs(y_par[i] - y_seq[i]));
            std::cout << std::fixed << std::setprecision(9)
                      << "col," << n << "," << procs << ","
                      << Tcol << "," << T1
                      << ", error=" << max_err << "\n";
        }

        // === Block-wise ===
        double Tblock = matvec_block(n, A_root, x_root, y_par, MPI_COMM_WORLD, repeats);
        if (rank == 0) {
            double max_err = 0.0;
            for (int i = 0; i < n; ++i)
                max_err = std::max(max_err, std::abs(y_par[i] - y_seq[i]));
            std::cout << std::fixed << std::setprecision(9)
                      << "block," << n << "," << procs << ","
                      << Tblock << "," << T1
                      << ", error=" << max_err << "\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
