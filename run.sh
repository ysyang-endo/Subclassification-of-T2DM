# 1. Train VAE model
# python vae.py --data log/data/clinical_data_vae.csv

# 2. Get VAE embedding
# python vae.py --data log/data/clinical_data_pretrain.csv --model_state_dict log/models/model_vae_100.pth
# python vae.py --data log/data/clinical_data_finetune.csv --model_state_dict log/models/model_vae_100.pth

# 3. Pretrain Transformer model
# python pretrain.py --data log/data/clinical_data_pretrain_vae.csv

# 4. Finetune Transformer model
# python finetune.py --data log/data/clinical_data_finetune_vae.csv --data_s log/data/clinical_data_finetune_s.csv --data_y log/data/clinical_data_finetune_y.csv --model_state_dict log/models/model_pretrain_clinical_500.pth

# 5. Get finetuning embedding
# python finetune.py --data log/data/clinical_data_finetune_vae.csv --data_s log/data/clinical_data_finetune_s.csv --data_y log/data/clinical_data_finetune_y.csv --model_state_dict log/models/model_pretrain_clinical_500.pth --finetune_model_state_dict log/models/model_finetune_clinical_300.pth