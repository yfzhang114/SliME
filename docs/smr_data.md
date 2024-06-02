
the SMR data structure

data
├── arxivqa
│   └── images
├── DVQA
│   └── images
├── Geometry3K
│   └── 0-2400 dirs
├── ChartQA
│   └── train_images
└── GeoQA3
│    ├── image
│    └── json
├── mathvision
├── scienceqa
├── tabmwp
└── GeoQA3
│    ├── train
│    └── test
│    └── val

Download images using this [download url](https://huggingface.co/datasets/MMInstruction/ArxivQA/resolve/main/images.tgz)

```python
dpo_utils/data_processing/process_arxivQA.py 
```

2. DVQA  

Download images using this [url](https://drive.google.com/file/d/1iKH2lTi1-QxtNUVRxTUWFvUvRHq6HAsZ/view?usp=sharing).

3. ChartQA

Clone this [repo](git@github.com:vis-nlp/ChartQA.git)

extract all the training images in ```ChartQA_Dataset/train/png``` into ```ChartQA```

4. Geometry3K

Download images using this [url](https://lupantech.github.io/inter-gps/geometry3k/train.zip).

The image path in our json file will be ```os.path.join(f'Geometry3K/i', 'img_diagram.png')```

5. GeoQA3

Download images using this [url](https://drive.google.com/drive/folders/1fiLTJUq7EPiZHs6AxundNfNEDLw4gtP5?usp=sharing)

extract all the training images in ```GeoQA3/image```

6. MathVision

Download images using this [url](https://github.com/mathvision-cuhk/MathVision/images)

Our data will not include the images from test-mini split automatically

7. ScienceQA

```bash
wget https://scienceqa.s3.us-west-1.amazonaws.com/images/train.zip
wget https://scienceqa.s3.us-west-1.amazonaws.com/images/val.zip
wget https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip

unzip -q train.zip
unzip -q val.zip
unzip -q test.zip

rm train.zip
rm val.zip
rm test.zip
```

8. Tabmwp

Download images using this [url](https://github.com/lupantech/PromptPG/tree/main/data/tabmwp/tables)

9. TextbookQA

Download images using this [url](https://ai2-public-datasets.s3.amazonaws.com/tqa/tqa_train_val_test.zip)
