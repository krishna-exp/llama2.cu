# llama2.cu

Check https://github.com/karpathy/llama2.c for reference implementation

## Testing

```bash
nvcc forward.cu testing.cu
./a.out
```

## Running a model
```bash
nvcc -arch=native run.cu forward.cu
./a.out <path to model.bin> -i "<input>" -t <temp>
```

## Profiling the code
```bash
nsys profile -o test.nsys-prof ./a.out <path to model.bin>
ncu --config-file off --kernel-name <kernel name> --export "report%i" --launch-count 1 --set full ./a.out
```
