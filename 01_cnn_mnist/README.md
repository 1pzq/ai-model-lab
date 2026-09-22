# 01 CNN MNIST

这是一个适合学习和 GPU 试跑的轻量 CNN。它会自动使用 CUDA；没有 GPU 时退回 CPU。

## 训练

在服务器上进入目录并激活虚拟环境：

```bash
cd ~/ai-model-lab/01_cnn_mnist
source ~/venvs/ai-model-lab-group/bin/activate
pip install -r requirements.txt
python train.py --epochs 5 --gpu 0
```

你的 `nvidia-smi` 中 GPU 0 是 RTX 3090，GPU 1 是 RTX A4000。使用 A4000 时：

```bash
python train.py --epochs 5 --gpu 1
```

训练数据会放到 `data/`，最佳模型保存到 `checkpoints/best.pt`。这些目录已在仓库的 `.gitignore` 中排除，不会提交到 GitHub。

## 查看 GPU

训练时另开一个终端执行：

```bash
nvidia-smi
```

## 预测

准备一张包含单个数字的图片后：

```bash
python predict.py /path/to/digit.png
```

## 推送到 GitHub

```bash
cd ~/ai-model-lab
git add 01_cnn_mnist
git commit -m "add CNN MNIST training example"
git push
```
