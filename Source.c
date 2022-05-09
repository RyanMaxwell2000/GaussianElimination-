#include <stdio.h>
#include <time.h>
#include <Windows.h>
//#include <omp.h>


// Declatations
// 
//Declare Matrix
float inputted_matrix[3][4] = {
{1,1,1,1},
{2,2,2,2},
{3,3,3,3}
};

float F[3][4] = {
{1,1,-1,-2},
{2,-1,1,5},
{-1,2,2,1}
};

float M[3][4] = {
{2,1,1,10},
{3,2,3,18},
{1,4,9,16}
};

float G[3][4] = {
{1, 1, -1, -3},
{ 6, 2, 2, 2 },
{ -3, 4, 1, 1 }
};

//Get Matrix size (Amount of Rows)   
#define matrix_size sizeof(inputted_matrix) / sizeof(inputted_matrix[0])


int main() {


    //omp_set_num_threads(4);
  
    //Start Timer
    clock_t t;
    t = clock();
    printf("Timer starts\n");

    // Forward Elimination:
    for (int k = 0; k < matrix_size - 1; k++) {
        
        

        //#pragma omp parallel shared(M, matrix_size, k, i) private(j)
        //#pragma omp for

        for (int i = k + 1; i < matrix_size; i++){
            
            
            //M[i][k] = M[i][k] / M[k][k];
            float multiple = M[i][k] / M[k][k];

            //#pragma omp parallel shared(M, matrix_size, k, i) private(j)
            //#pragma omp for

            for (int j = k; j < matrix_size; j++){
                //M[i][j] = M[i][j] - M[i][k] * M[k][j];
                M[i][j] = M[i][j] - multiple * M[k][j];

            }
            //M[i][matrix_size] = M[i][matrix_size] - M[i][k] * M[k][matrix_size];
            M[i][matrix_size] = M[i][matrix_size] - multiple * M[k][matrix_size];
        }
    }

    //Calculating Value
    float outputs[matrix_size];
    for (int i = 0; i < matrix_size; i++) {
        outputs[i] = 0;
    }

    outputs[matrix_size-1] = M[matrix_size - 1][matrix_size] / M[matrix_size - 1][matrix_size-1];

    for (int i = matrix_size-1; i >= 0; i--) {

        float sum = M[i][matrix_size];
        for (int j = i +1 ; j < matrix_size; j++) {
            sum = sum - ( (M[i][j] * outputs[j]));

        }
        outputs[i] = sum / (M[i][i]);
    }


    // result of X
    for (int i = 0; i < matrix_size; i++)
    {
        printf("X%d Value: %.2f \n", i + 1, outputs[i]);
    }

    printf("\n");
    printf("\n");

    // calculate the elapsed time
    t = clock() - t;
    double time_taken = ((double)t) / CLOCKS_PER_SEC;
    printf("The program took %f seconds to execute", time_taken);

    return 0;
}