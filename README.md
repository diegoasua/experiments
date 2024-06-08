# experiments

## Proving CUDA is Turing complete
GPUs are Turing complete. So is CUDA, as a general GPU programming language. Let's simulate a Turing universal machine in CUDA:

To compile `turing_machine.cu`:
```bash
nvcc -o turing_machine turing_machine.cu
```

If you want to simulate this in a Colab instance you can add the line `%%writefile turing_machine.cu` at the beginning of the cell to write the file to directory and use the special escape symbol `!` to execute cells as bash.
