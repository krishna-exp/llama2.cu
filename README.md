# llama2.cu

Check https://github.com/karpathy/llama2.c for reference implementation

## Testing

```bash
nvcc forward.cu testing.cu
./a.out
```

## Running a model
```bash
nvcc run.cu forward.cu
./a.out <path to model.bin> -i "<input>" -t <temp>
```

## Profiling the code
```bash
nsys profile -o test.nsys-prof ./a.out <path to model.bin>
```
