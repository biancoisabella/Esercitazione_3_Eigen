#include <iostream>
#include <iomanip>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;



double get_relative_error(VectorXd my_sol, VectorXd sol){
    return (my_sol-sol).norm()/sol.norm();
}


VectorXd PALU_solver(MatrixXd &A, VectorXd &b, double &rel_error, VectorXd sol = VectorXd{{-1.0e+0}, {-1.0e+00}}){
    VectorXd x = A.lu().solve(b);
    rel_error = get_relative_error(x, sol);
    return x;
}


VectorXd QR_solver(MatrixXd &A, VectorXd &b, double &rel_error, VectorXd sol = VectorXd{{-1.0e+0}, {-1.0e+00}}){
    VectorXd x = A.householderQr().solve(b);
    rel_error = get_relative_error(x, sol);
    return  x;
}

void output_problem(MatrixXd A, VectorXd b, string problem_name);



int main()
{
    // Scelgo il corretto formato di stampa sul file di output
    cout << std::setprecision(16) << std::scientific;

    output_problem(MatrixXd{{5.547001962252291e-01, -3.770900990025203e-02},{8.320502943378437e-01, -9.992887623566787e-01}},
        VectorXd{{-5.169911863249772e-01}, {1.672384680188350e-01}}, "PRIMO");



    output_problem(MatrixXd{{5.547001962252291e-01, -5.540607316466765e-01}, {8.320502943378437e-01, -8.324762492991313e-01}},
        VectorXd{{-6.394645785530173e-04}, {4.259549612877223e-04}}, "SECONDO");

    output_problem(MatrixXd{{5.547001962252291e-01, -5.547001955851905e-01}, {8.320502943378437e-01,-8.320502947645361e-01}},
        VectorXd{{-6.400391328043042e-10}, {4.266924591433963e-10}}, "TERZO");


    return 0;
}

void output_problem(MatrixXd A, VectorXd b, string problem_name){
    double err_relativo = 0.0;

    cout << "---------------- " << problem_name << " PROBLEMA ----------------- " << endl;
    cout << "Matrice A: " << endl;
    cout << A << endl;
 
    cout << "Vettore b " << endl;
    cout << b << endl << endl;
 
    cout << "Soluzione della fattorizzazione PA = LU: " << endl;
    cout << PALU_solver(A,b, err_relativo) << endl;
    cout << "Errore della soluzione con fattorizzazione PA = LU: " << err_relativo << endl<<endl;
 
    cout << "Soluzione della fattorizzazione A = QR: " << endl;
    cout << QR_solver(A,b, err_relativo) << endl;
    cout << "Errore della soluzione con fattorizzazione A = QR: " << err_relativo << endl;
    
    cout << "------------------------------------------------- " << endl << endl;
    return;
}
