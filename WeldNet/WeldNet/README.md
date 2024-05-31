1. Before running WeldNet, you need to install the environment in requirements.txt.

2. Data set preparation

The format of the data set is COCO format and is located in the .datasets\coco128 folder

3. WeldNet training

Run the train.py file. The specific instructions are as follows:
python train.py --img 640 --batch 2 --epochs 400 --data ./data/coco128.yaml --cfg ./models/yolov5s_ShuffleCBAM.yaml --weights ''

4. Training results

The training results are located at .runs\train\exp folder

5. WeldNet testing

Run the detect.py file. The specific instructions are as follows:
python detect.py

6. Test results

The test results are located in the .runs\detect\exp  file

7. Citing
If you use this project's code for your academic work, we encourage you to cite our papers

Y. Ma, J. Fan, S. Zhao, F. Jing, S. Wang and M. Tan, "From Model to Reality: A Robust Framework for Automatic Generation of Welding Paths," in IEEE Transactions on Industrial Electronics, doi: 10.1109/TIE.2024.3395792.

8. Reference resources：

   [ultralytics/yolov5](https://ultralytics.com/yolov5)
   [CBAM.PyTorch](https://github.com/luuuyi/CBAM.PyTorch)
   [ShuffleNetV2-pytorch](https://github.com/Randl/ShuffleNetV2-pytorch)
