# README #


### What is this repository for? ###

* OXCNN_core is a barebones python program designed to be extended to create more complex neural networks for patch based image analysis
* Version 0.01
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up
Create a set of test cases with 

python3 -m oxcnn.test.utils ~/Desktop/TestVolumes

Write out the TensorFlowRecords with

python3 main.py --model models.deepmedic write --save_dir ~/Desktop/TestRect-tfr --data_dir ~/Desktop/TestRect/

Train your the model with

python3 main.py --model models.deepmedic train --tfr_dir ~/Desktop/TestRect-tfr --save_dir ~/Desktop/TestRect-out --test_data ~/Desktop/TestRect-tfr/meta_data.txt --num_epochs 10 --batch_size 10

Test your model with 

python3 main.py --model models.deepmedic test --save_dir ~/Desktop/TestRect-out/test --test_data_file ~/Desktop/TestRect/meta_data.txt --model_file ~/Desktop/TestRect-out/model.ckpt-13024.meta



* Configuration

* Dependencies
TFLearn v0.3, Tensorflow v1.1, Pandas

* How to run tests
python3 -m unittest

* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact