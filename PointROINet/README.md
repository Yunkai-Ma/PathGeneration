1. Before running PointROINet, you need to install the environment in requirements.txt.

2. Data set preparation
The format of the data set is ShapeNet format and is located in the .data\shapenetcore_partanno_segmentation_benchmark_v0_normal folder

3. PointROINet training
Run the train_partseg.py file. The specific instructions are as follows:
python train_partseg.py --model ppointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg

4. Training results
The training results are located at .log\part_seg\pointnet2_part_seg_msg folder

5. PointROINet testing
Run the test_partseg_showweldpointcloud.py file. The specific instructions are as follows:
python test_partseg_showweldpointcloud.py --normal --log_dir pointnet2_part_seg_msg

6. Test results
The test results are located in the log\part_seg\pointnet2_part_seg_msg\eval.txt file

7. Reference resources：
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)<br>
[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)<br>
[charlesq34/PointNet](https://github.com/charlesq34/pointnet) <br>
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)
[Xu/Pytorch_Pointnet_Pointnet2](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)