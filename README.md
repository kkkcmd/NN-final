# NN-final
Vit.py为Transformer架构下图像分类算法的实现过程，Resnet18.py为CNN架构下图像分类算法的实现过程。

Vit-tiny模型参数的下载路径为：
https://huggingface.co/WinKawaks/vit-tiny-patch16-224/tree/main
需要将以上路径中的文件下载到本地，并保存在同一目录下vit-tiny-patch16-224文件夹中

Resnet模型无需提前下载模型。

在安装好对应包的python环境中，直接通过以下命令可以运行上述函数：

python Vit.py

python Resnet18.py

可以通过超参数列表来设置参数。

运行tensorboard --logdir=./runs可以查看训练过程，仓库中已经提前给出两个模型的记录文件，直接将runs文件夹进行替换即可。

如果在百度云网盘中提前下载完成训练好的模型权重文件，则权重文件的加载代码为: （以vit模型为例）

vit_model = ...  # 创建与保存时相同的模型实例

model_load_path = 'vit_model_weights.pth'

vit_model.load_state_dict(torch.load(model_load_path))

vit_model.eval()
