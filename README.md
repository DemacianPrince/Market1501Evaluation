# Market-1501 Evaluation Code
Matlab codes for evaluating performance on [Market-1501 Person Re-ID Dataset](http://www.liangzheng.com.cn/Project/project_reid.html).

### Usage
1. Calculate and save the feature for the query and gallery images of the dataset in advance;

2. Change the query and gallery feature directory in the codes correspondingly;

3. Run `market_evaluation.m` to get the Rank1-Accuracy and mAP result.

### Citation
If you use this code, please kindly cite this paper:

     @inproceedings{zheng2015scalable,
       title={Scalable Person Re-identification: A Benchmark},
       author={Zheng, Liang and Shen, Liyue and Tian, Lu and Wang, Shengjin and Wang, Jingdong and Tian, Qi},
       booktitle={Computer Vision, IEEE International Conference on},
       year={2015}
     }
