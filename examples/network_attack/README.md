```bash
fado -f config/fado_config.yaml
```

In order to implement the attack the following steps were done:
1) Write the model definition (initially written for TF) for PyTorch
2) Implement code for downloading and transforming the data to be read by FADO
3) Make sure that the FL process is behaving the same (same parameters)
4) 