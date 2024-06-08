#include <iostream>
#include <cuda_runtime.h>

#define STATE_INITIAL 0
#define STATE_FINAL 1
#define HALT_STATE -1
#define TAPE_SIZE 100

__global__ void turingMachineSimulator(char *tape, int *state, int *headPos) {
    while (*state != HALT_STATE) {
        char currentSymbol = tape[*headPos];
        switch (*state) {
            case STATE_INITIAL:
                if (currentSymbol == '1') {
                    tape[*headPos] = '0';
                    (*headPos)++;
                    *state = STATE_FINAL;
                } else {
                    *state = HALT_STATE;
                }
                break;
            case STATE_FINAL:
                if (currentSymbol == '0') {
                    tape[*headPos] = '1';
                    (*headPos)--;
                    *state = HALT_STATE;
                } else {
                    *state = HALT_STATE;
                }
                break;
            default:
                *state = HALT_STATE;
        }
    }
}

int main() {
    char h_tape[TAPE_SIZE];
    int h_state = STATE_INITIAL;
    int h_headPos = 0;

    for (int i = 0; i < TAPE_SIZE; i++) {
        h_tape[i] = (i % 2 == 0) ? '1' : '0';
    }

    char *d_tape;
    int *d_state, *d_headPos;
    cudaMalloc(&d_tape, TAPE_SIZE * sizeof(char));
    cudaMalloc(&d_state, sizeof(int));
    cudaMalloc(&d_headPos, sizeof(int));

    cudaMemcpy(d_tape, h_tape, TAPE_SIZE * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_state, &h_state, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_headPos, &h_headPos, sizeof(int), cudaMemcpyHostToDevice);

    turingMachineSimulator<<<1, 1>>>(d_tape, d_state, d_headPos);

    cudaMemcpy(h_tape, d_tape, TAPE_SIZE * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_state, d_state, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_headPos, d_headPos, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Final tape: ";
    for (int i = 0; i < TAPE_SIZE; i++) {
        std::cout << h_tape[i];
    }
    std::cout << std::endl;

    std::cout << "Final state: " << h_state << std::endl;
    std::cout << "Final head position: " << h_headPos << std::endl;

    cudaFree(d_tape);
    cudaFree(d_state);
    cudaFree(d_headPos);

    return 0;
}
