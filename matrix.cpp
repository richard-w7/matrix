#include <iostream>
#include <vector>
#include <stdexcept>
#include <cassert>

template <typename T>
class Matrix {
private:
    size_t rows_;
    size_t cols_;
    std::vector<T> data_;

public:
    // Constructors
    Matrix(size_t rows, size_t cols) 
        : rows_(rows), cols_(cols), data_(rows * cols) {}

    Matrix(size_t rows, size_t cols, const std::vector<T>& values)
        : rows_(rows), cols_(cols), data_(values) {
        assert(values.size() == rows * cols);
    }

    // Accessors
    T& operator()(size_t row, size_t col) {
        return data_[row * cols_ + col];
    }

    const T& operator()(size_t row, size_t col) const {
        return data_[row * cols_ + col];
    }

    // Addition
    Matrix operator+(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }
        
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }

    // Multiplication (cache-optimised)
    Matrix operator*(const Matrix& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument(
                "Matrix dimensions must match for multiplication");
        }

        Matrix transposed = other.transpose();
        Matrix result(rows_, other.cols_);

        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < other.cols_; ++j) {
                T sum = 0;
                for (size_t k = 0; k < cols_; ++k) {
                    sum += (*this)(i, k) * transposed(j, k);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    // Transpose
    Matrix transpose() const {
        Matrix result(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    // Determinant with laplace expansion
    T determinant() const {
        if (rows_ != cols_) {
            throw std::invalid_argument("Matrix must be square for determinant");
        }

        if (rows_ == 1) return data_[0];
        if (rows_ == 2) {
            return data_[0] * data_[3] - data_[1] * data_[2];
        }

        T det = 0;
        for (size_t col = 0; col < cols_; ++col) {
            Matrix minor = create_minor(0, col);
            T sign = (col % 2 == 0) ? 1 : -1;
            det += sign * data_[col] * minor.determinant();
        }
        return det;
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

private:
    // Create minor matrix by removing specified row and column
    Matrix create_minor(size_t row, size_t col) const {
        Matrix minor(rows_ - 1, cols_ - 1);
        size_t minor_row = 0;
        
        for (size_t i = 0; i < rows_; ++i) {
            if (i == row) continue;
            size_t minor_col = 0;
            
            for (size_t j = 0; j < cols_; ++j) {
                if (j == col) continue;
                minor(minor_row, minor_col) = (*this)(i, j);
                ++minor_col;
            }
            ++minor_row;
        }
        return minor;
    }
};

// printing
template <typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& m) {
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            os << m(i, j) << "\t";
        }
        os << "\n";
    }
    return os;
}

int main() {
    try {
        // Create matrices
        Matrix<int> a(2, 2, {1, 2, 3, 4});
        Matrix<int> b(2, 2, {5, 6, 7, 8});
        Matrix<int> c(3, 3, {2, -3, 1, 2, 0, -1, 1, 4, 5});

        // Test operations
        std::cout << "Matrix A:\n" << a << "\n";
        std::cout << "Matrix B:\n" << b << "\n";
        std::cout << "A + B:\n" << a + b << "\n";
        std::cout << "A * B:\n" << a * b << "\n";
        std::cout << "Transpose of A:\n" << a.transpose() << "\n";
        std::cout << "Determinant of C: " << c.determinant() << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}