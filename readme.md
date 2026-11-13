
conda create -n train_llm python=3.11
conda activate train_llm
pip install uv

pip install  --no-build-isolation  -r requirements.txt


可以在自定义数据集和swift支持数据集训练
样例trainingexample中训练模型
cd trainingexample
修改config
python traininig.py
