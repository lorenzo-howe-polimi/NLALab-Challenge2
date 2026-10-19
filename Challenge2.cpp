#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <unsupported/Eigen/SparseExtra>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cmath>
#include <Eigen/SVD>

using Matrix = Eigen::MatrixXd;
using ImageMatrix =
	Eigen::Matrix<unsigned char,
				  Eigen::Dynamic,
				  Eigen::Dynamic,
				  Eigen::RowMajor>;

using ImageMap = Eigen::Map<ImageMatrix>;
using Vector = Eigen::VectorXd;
using Eigen::RowMajor;
using SparseMatrix = Eigen::SparseMatrix<double>;

ImageMatrix toImage(const Matrix &matrix)
{
	return matrix.unaryExpr([](double value)
							{ return static_cast<unsigned char>(
								  std::clamp(value, 0.0, 255.0)); });
}

bool saveImage(const std::string &path, const ImageMatrix &image, int width, int height)
{
	return stbi_write_png(path.c_str(), width, height, 1, image.data(), width) != 0;
}

/*		ACTUAL IMPORTANT CODE STARTS HERE	*/
int main(int argc, char *argv[])
{
	int width, height, channels;
	unsigned char *image_data;
	Matrix ATA;
	Matrix A;
	Matrix C;
    Matrix D;
    Matrix CDT;
	Matrix NC;
    Matrix Sigma;
	ImageMatrix image;
    int k;

	if (argc < 2)
	{
		return std::cerr << "Usage: " << argv[0] << " <image_path>\n", 1;
	}
	// TASK 1--------------------------------------------------------------------------
	image_data = stbi_load(argv[1], &width, &height, &channels, 1);
	if (!image_data){return std::cerr << "Error: Could not load image " << argv[1] << '\n', 1;}
	A = ImageMap(image_data, height, width).cast<double>();
	stbi_image_free(image_data);
	ATA = A.transpose() * A;
	std::cout << "ATA norm: " << ATA.norm() << std::endl;
	// TASK 2--------------------------------------------------------------------------
	Eigen::SelfAdjointEigenSolver<Matrix> solver(ATA, Eigen::EigenvaluesOnly);
	if (solver.info() != Eigen::Success) abort();
	const auto &ev = solver.eigenvalues();
	const auto n = ev.size();
	std::cout << "The largest eigen value: " << std::sqrt(ev(n - 1)) << '\n' << "The second largest eigen value: " << std::sqrt(ev(n - 2)) << '\n';
	// TASK 3--------------------------------------------------------------------------
	if (!Eigen::saveMarket(ATA, "output_data/ATA.mtx")){return std::cerr << "Error: Could not save ATA.mtx\n", 1;};
	// mpirun -n 4 ./eigen1 output_data/ATA.mtx eigenval.txt hist.txt -e pi -etol 1.e-10  (to run in terminal)
	// TASK 4--------------------------------------------------------------------------
	std::cout << "mu shift: " << (ev(n - 2) + ev(0)) / 2 << std::endl;
	// mpirun -n 4 ./eigen1 output_data/ATA.mtx eigenval.txt hist.txt -e pi -etol 1.e-10 -shift 5.64322e07 (to run in terminal)
    // TASK 5 -------------------------------------------------------------------------
    Eigen::BDCSVD<Eigen::MatrixXd> svd (A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Sigma = svd.singularValues().asDiagonal();
    std::cout <<"Sigma norm: "<< Sigma.norm()<< std::endl;
    // TASK 6 -----------------------------------------------------------------------
    k = 90;     // k= 30
    C = svd.matrixU().leftCols(k);
    D = svd.matrixV().leftCols(k)*Sigma.leftCols(k).topRows(k);
    std::cout<<"NNz elements in C: "<<C.nonZeros()<<std::endl<<"Nnz elements in D: "<<D.nonZeros()<< std::endl;
    // TASK 7 -----------------------------------------------------------------------
    CDT = C*D.transpose();
    image = toImage(CDT);
	if (!saveImage("output_images/CompressedImage30.png", image, 656, 656)){return std::cerr << "Error: Could not save image\n", 1;}
	// TASK 8--------------------------------------------------------------------------
	C = Matrix::Zero(400, 400);

	for (int i = 0; i < 400; ++i)
		for (int j = 0; j < 400; ++j)
			if ((i + j) % 2 == 1)
				C(i, j) = 255;
	std::cout << "Euclidean norm: " << C.norm() << '\n';
	// TASK 9--------------------------------------------------------------------------
	NC = C + 60.0 * Matrix::Random(400, 400);
	image = toImage(NC);
	if (!saveImage("output_images/noisy_checkerboard_image.png", image, 400, 400)){return std::cerr << "Error: Could not save image\n", 1;}
	std::cout << "Noisy checkerboard image saved to noisy_checkerboard_image.png" << std::endl;
	return 0;
}