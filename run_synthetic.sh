python3 run_synthetic.py --model linear --name squeeze_ --latent_dim 3
python3 plot_synthetic.py --input_file output/squeeze_linear_losses.csv --output_prefix squeeze_linear --linear
python3 run_synthetic.py --model relu --name squeeze_ --latent_dim 3
python3 plot_synthetic.py --input_file output/squeeze_relu_losses.csv --output_prefix squeeze_relu
python3 run_synthetic.py --model tanh --name squeeze_ --latent_dim 3
python3 plot_synthetic.py --input_file output/squeeze_tanh_losses.csv --output_prefix squeeze_anh
