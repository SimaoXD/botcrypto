para iniciar:

pip install 
venv\Scripts\activate  
python src/main.py

Se quiser desativar essas otimizações, defina essa variável de ambiente como 0 antes de rodar o script:
venv\Scripts\activate  
 $env:TF_ENABLE_ONEDNN_OPTS=0
python src/main2.py
