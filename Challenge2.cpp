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

ImageMatrix toImage(const Matrix& matrix)
{
	return matrix.unaryExpr([](double value) {
		return static_cast<unsigned char>(
			std::clamp(value, 0.0, 255.0)
		);
	});
}

bool saveImage(const std::string& path, const ImageMatrix& image, int width, int height)
{
	return stbi_write_png(path.c_str(), width, height, 1, image.data(), width) != 0;
}


/*		ACTUAL IMPORTANT CODE STARTS HERE	*/
int	main(int argc, char *argv[])
{
	int				width, height, channels;
	unsigned char	*image_data;
    Matrix ATA;
    Matrix A;

    //task1

	if (argc < 2){ return std::cerr << "Usage: " << argv[0] << " <image_path>\n", 1;}
	//TASK 1 START--------------------------------------------------------------------------
	image_data = stbi_load(argv[1], &width, &height, &channels, 1);
	if (!image_data) {return std::cerr << "Error: Could not load image " << argv[1] << '\n', 1;}
	A = ImageMap(image_data, height, width).cast<double>();
	stbi_image_free(image_data);

    ATA = A.transpose()*A;

    std::cout << "ATA norm: " << ATA.norm() << std::endl;

    //task2
   Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(ATA);    
   if (eigensolver.info() != Eigen::Success) abort();
   auto const& ev = eigensolver.eigenvalues();
   int n = ev.size();
   std::cout<< "The 2 largest singular values of A are:\n" << ev.tail(2) << std::endl;
   std::cout<< "The smallest singular value of A is:\n" << ev.head(1) << std::endl;
   //std::cout << "The 2 largest singular values of A are:\n" << eigensolver.eigenvalues().tail(2) << std::endl;
   // std::cout << "Here's a matrix whose columns are eigenvectors of A \n"
   //           << eigensolver.eigenvectors() << std::endl;
   std::cout << "mu shift" <<(ev(n-2)+ev(0))/2 << std::endl;

   //task3
   if (!Eigen::saveMarket(ATA, "ATA.mtx")){return std::cerr << "Error: Could not save ATA.mtx\n", 1;};
   return 0;
}